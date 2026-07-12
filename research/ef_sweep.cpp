// ef_sweep.cpp
// For each dataset:
//   1. One-time probe: build EfAdapter with ef_upper_bound=5000 to discover
//      the dataset's natural ef cap (max ef stored in the smoothed table).
//   2. Sweep ef_max = 100, 150, ..., natural_cap in steps of 50.
//      At each step rebuild EfAdapter + CV buckets, run adaptive search.
//   3. Early-stop when WAE growth between steps falls below WAE_MIN_GROWTH.
//   4. Emit: RESULT|<dataset>|<ef_max>|<avg_recall>|<latency_ms>
//
// Build: cmake --build build/ --target ef_sweep -j$(nproc)
// Run:   EXPERIMENTS_ROOT=/path/to/root ./build/ef_sweep > research/ef_sweep.log
// Plot:  python3 research/plot_ef_sweep.py research/ef_sweep.log

#include "../experiments_driver/util.h"
#include "../hnswlib/adaptive_ef.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct DatasetConfig {
    std::string name;
    std::string metric;
    size_t      k;
    float       alpha;
    float       gamma;
    float       expected_recall;
};

static const std::vector<DatasetConfig> DATASETS = {
    {"deep-image-96-angular", "cd", 100, 0.25f, 16.0f, 0.95f},
    {"glove-100-angular",     "cd", 100, 0.25f, 16.0f, 0.95f},
    {"sift-128-euclidean",    "l2", 100, 0.25f, 16.0f, 0.95f},
};

static const size_t STATICS_LENGTH = 1 + 32 + 31 * 32; // 2-hop on base layer, M=16
static const int    N_DEP_TABLES   = 10;
static const int    REPEAT         = 3;
static const int    SAMPLING_SIZE  = 2000;

static constexpr int    EF_PROBE    = 5000;   // upper bound for the one-time probe
static constexpr int    EF_START    = 100;
static constexpr int    EF_STEP     = 50;
static constexpr double WAE_MIN_GROWTH = 0.0101; // stop when WAE grows < 1.01%

// Returns the highest ef value stored in any row of the smoothed table.
// This is the natural dataset cap: the largest ef the table-builder needed
// to satisfy expected_recall given ef_upper_bound=EF_PROBE.
static int probe_natural_cap(
    const std::shared_ptr<hnswlib::HierarchicalNSW<float>>& hnsw,
    const std::shared_ptr<hnswdis::MatrixXf>&               data,
    const std::shared_ptr<hnswdis::MatrixXf>&               sample_queries_ptr,
    const std::shared_ptr<hnswdis::MatrixXi>&               sample_gt_ptr,
    const DatasetConfig&                                    ds)
{
    hnswdis::EfAdapter probe(
        hnsw, data,
        ds.k, ds.metric, ds.expected_recall,
        ds.alpha, ds.gamma,
        STATICS_LENGTH,
        sample_queries_ptr, sample_gt_ptr,
        EF_PROBE
    );

    int cap = EF_START;
    for (const auto& [score, ef_recall_list] : probe.get_ef_recall_estimators())
        for (const auto& [ef, recall] : ef_recall_list)
            cap = std::max(cap, ef);

    return cap;
}

static hnswdis::Sketch make_sketch(const hnswdis::EfAdapter& adapter, float expected_recall) {
    if (adapter.has_cv_tables())
        return hnswdis::Sketch(adapter.get_all_tables(), adapter.get_cv_centers(), expected_recall);
    return hnswdis::Sketch(adapter.get_ef_recall_estimators(), expected_recall);
}

int main() {
    const char* root_env = std::getenv("EXPERIMENTS_ROOT");
    if (!root_env) {
        std::cerr << "Error: EXPERIMENTS_ROOT is not set.\n"
                  << "  export EXPERIMENTS_ROOT=/path/to/experiments\n";
        return 1;
    }
    const fs::path root(root_env);

    Eigen::setNbThreads(std::max(1u, std::thread::hardware_concurrency() / 4));

    std::cout << "# ef_sweep log\n"
              << "# format: RESULT|dataset|ef_max|avg_recall|latency_ms\n"
              << "# EXPERIMENTS_ROOT=" << root_env << "\n"
              << "# ef_probe=" << EF_PROBE
              << "  ef_start=" << EF_START
              << "  ef_step=" << EF_STEP
              << "  wae_min_growth=" << WAE_MIN_GROWTH << "\n";

    for (const auto& ds : DATASETS) {
        const std::string hdf5_path  = (root / "data"  / (ds.name + ".hdf5")).string();
        const std::string index_path = (root / "index" / (ds.name + "-M16-efc-500-parallel.hnsw")).string();

        std::cerr << "\n========== " << ds.name << " ==========\n";

        std::shared_ptr<hnswlib::HierarchicalNSW<float>>  hnsw;
        std::shared_ptr<hnswdis::MatrixXf>                query;
        std::shared_ptr<hnswdis::MatrixXf>                data;
        std::shared_ptr<hnswdis::MatrixXi>                ground_truth;
        std::shared_ptr<hnswlib::SpaceInterface<float>>   space;

        try {
            auto tup    = load_index_and_data(hdf5_path, index_path, ds.metric);
            hnsw        = std::get<0>(tup);
            query       = std::get<1>(tup);
            data        = std::get<2>(tup);
            ground_truth= std::get<3>(tup);
            space       = std::get<4>(tup);
        } catch (const std::exception& e) {
            std::cerr << "  [SKIP] " << e.what() << "\n";
            continue;
        }

        // Samplings computed once per dataset; reused across all ef_max values
        std::cerr << "  Computing samplings (size=" << SAMPLING_SIZE << ")...\n";
        auto [sample_queries, sample_gt] =
            hnswdis::compute_samplings(data, ds.metric, ds.k, SAMPLING_SIZE);
        auto sample_queries_ptr = std::make_shared<hnswdis::MatrixXf>(sample_queries);
        auto sample_gt_ptr      = std::make_shared<hnswdis::MatrixXi>(sample_gt);

        hnswdis::ApproximatedScoreCalculator score_cal(ds.alpha);

        // One-time probe: build with EF_PROBE to find where the table naturally saturates
        std::cerr << "  Probing natural ef cap (ef_upper_bound=" << EF_PROBE << ")...\n";
        const int natural_cap = probe_natural_cap(hnsw, data, sample_queries_ptr, sample_gt_ptr, ds);
        std::cerr << "  Natural cap for " << ds.name << ": ef=" << natural_cap << "\n";
        std::cout << "# CAP|" << ds.name << "|" << natural_cap << "\n" << std::flush;

        float prev_wae = 0.0f;
        int   ef_max   = EF_START;

        while (ef_max <= natural_cap) {
            std::cerr << "  ef_max=" << ef_max << " building lookup table...\n";

            // Rebuild EfAdapter with this ef_upper_bound
            hnswdis::EfAdapter adapter(
                hnsw, data,
                ds.k, ds.metric, ds.expected_recall,
                ds.alpha, ds.gamma,
                STATICS_LENGTH,
                sample_queries_ptr, sample_gt_ptr,
                ef_max
            );

            // Train 2-D CV-bucket table (shiro-ef)
            adapter.init_with_cv_buckets(
                hnsw, data,
                ds.k, ds.metric, ds.alpha, ds.gamma,
                STATICS_LENGTH,
                sample_queries_ptr, sample_gt_ptr,
                N_DEP_TABLES
            );

            hnswdis::Sketch sketch = make_sketch(adapter, ds.expected_recall);
            const float wae = adapter.get_wae();
            hnsw->setEf(static_cast<size_t>(wae));

            // Early-stop: WAE grew less than WAE_MIN_GROWTH relative to previous step
            if (prev_wae > 0.0f) {
                const double growth = (wae - prev_wae) / prev_wae;
                std::cerr << "    WAE=" << (size_t)wae
                          << "  growth=" << (growth * 100.0) << "%\n";
                if (growth < WAE_MIN_GROWTH) {
                    std::cerr << "  [STOP] WAE growth " << (growth * 100.0)
                              << "% < " << (WAE_MIN_GROWTH * 100.0)
                              << "% threshold at ef_max=" << ef_max << "\n";
                    break;
                }
            } else {
                std::cerr << "    WAE=" << (size_t)wae << "\n";
            }
            prev_wae = wae;

            const int n_queries = static_cast<int>(query->rows());
            std::vector<int64_t> times;
            times.reserve(REPEAT);
            std::vector<std::vector<size_t>> last_result;

            for (int r = 0; r < REPEAT; ++r) {
                std::vector<std::vector<size_t>> result(n_queries,
                    std::vector<size_t>(ds.k, 0));

                auto t0 = std::chrono::high_resolution_clock::now();
                for (int j = 0; j < n_queries; ++j) {
                    auto pq = hnsw->adaptiveSearchKnnTest(
                        query->row(j).data(), ds.k,
                        STATICS_LENGTH, score_cal, &sketch);
                    size_t cnt = pq.size();
                    while (!pq.empty()) {
                        result[j][--cnt] = pq.top().second;
                        pq.pop();
                    }
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                times.push_back(
                    std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

                if (r == REPEAT - 1)
                    last_result = std::move(result);
            }

            std::sort(times.begin(), times.end());
            const int64_t median_ms = times[times.size() / 2];

            const auto recalls =
                hnswdis::compute_recall(*ground_truth, last_result, ds.k, false);
            const double avg_recall =
                std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();

            std::cout << "RESULT|" << ds.name << "|" << ef_max
                      << "|" << avg_recall
                      << "|" << median_ms
                      << "\n" << std::flush;

            std::cerr << "    recall=" << avg_recall
                      << "  latency=" << median_ms << "ms\n";

            ef_max += EF_STEP;
        }
    }

    std::cerr << "\nDone.\n";
    return 0;
}
