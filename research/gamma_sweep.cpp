// gamma_sweep.cpp
// Sweep gamma (decay constant) across all 3 datasets.
// For each (dataset, gamma) pair:
//   - Loads existing samplings (pre-built by offline_exp)
//   - Rebuilds EfAdapter in-process with the swept gamma
//   - Runs adaptive_search (repeat=3, takes median)
//   - Emits a structured GAMMA_SWEEP log line for the Python visualizer
//
// All other parameters (alpha, k, ef_upper_bound, expected_recall) are held
// constant at their default values from g_experiments.
//
// Compile via: cmake --build build --target gamma_sweep
// Run via:     EXPERIMENTS_ROOT=/path/to/root ./build/gamma_sweep 2>&1 | tee gamma_sweep.log

#include "../experiments_driver/util.h"
#include "../hnswlib/adaptive_ef.h"
#include <filesystem>
#include <cstdlib>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <tuple>

// ============================================================================
// Gamma values to sweep
// ============================================================================
static const std::vector<float> GAMMA_VALUES = {
    1.0f, 2.0f, 4.0f, 8.0f, 12.0f, 16.0f, 24.0f, 32.0f, 48.0f, 64.0f
};

// ============================================================================
// Dataset configs — all params EXCEPT gamma are held constant
// ============================================================================
struct DatasetConfig {
    std::string dataset;
    std::string metric;
    size_t      k;
    float       alpha;
    float       expected_recall;
    int         ef_upper_bound;
    int         repeat;
};

static const std::vector<DatasetConfig> DATASETS = {
    {"deep-image-96-angular", "cd",  100, 0.25f, 0.95f, 4000, 3},
    {"glove-100-angular",     "cd",  100, 0.25f, 0.95f, 4000, 3},
    {"sift-128-euclidean",    "l2",  100, 0.25f, 0.95f,  350, 3},
};

// ============================================================================
// Helpers
// ============================================================================
static constexpr int N_DEP_TABLES = 10;

static hnswdis::Sketch make_sketch(const hnswdis::EfAdapter& adapter, float expected_recall) {
    if (adapter.has_cv_tables())
        return hnswdis::Sketch(adapter.get_all_tables(), adapter.get_cv_centers(), expected_recall);
    return hnswdis::Sketch(adapter.get_ef_recall_estimators(), expected_recall);
}

// Retrain CV buckets on top of a freshly-built EfAdapter.
static void train_cv_buckets(
    hnswdis::EfAdapter& adapter,
    const std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw,
    const std::shared_ptr<hnswdis::MatrixXf>               data,
    size_t k, const std::string& metric,
    float alpha, float gamma, size_t statics_length,
    const std::shared_ptr<hnswdis::MatrixXf> query_vectors,
    const std::shared_ptr<hnswdis::MatrixXi> ground_truth,
    int ef_upper_bound)
{
    adapter.init_with_cv_buckets(
        hnsw, data, k, metric, alpha, gamma, statics_length,
        query_vectors, ground_truth,
        N_DEP_TABLES);
}

// Run the adaptive search repeat times, return (median_ms, avg_recall, p5_recall, p1_recall, wae).
static std::tuple<int64_t, double, float, float, float> run_sweep(
    const std::string&                         dataset,
    int                                        repeat,
    hnswlib::HierarchicalNSW<float>&           hnsw,
    const hnswdis::MatrixXf&                   query,
    const hnswdis::MatrixXf&                   data,
    const hnswdis::MatrixXi&                   ground_truth,
    const hnswdis::ApproximatedScoreCalculator& score_cal,
    size_t k,
    hnswdis::Sketch&                           sketch,
    size_t statics_length,
    float expected_recall,
    float wae)
{
    hnsw.setEf(static_cast<size_t>(wae));

    std::vector<int64_t> times;
    times.reserve(repeat);
    double avg_recall = 0.0;
    float p5_recall = 0.0f;
    float p1_recall = 0.0f;

    for (int i = 0; i < repeat; ++i) {
        std::vector<std::vector<size_t>> result(query.rows(), std::vector<size_t>(k, 0));

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < query.rows(); ++j) {
            auto pq = hnsw.adaptiveSearchKnnTest(
                query.row(j).data(), k, statics_length, score_cal, &sketch);
            size_t cnt = pq.size();
            while (!pq.empty()) {
                result[j][--cnt] = pq.top().second;
                pq.pop();
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

        if (i == repeat - 1) {
            auto recalls = hnswdis::compute_recall(ground_truth, result, k, false);
            avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
            std::sort(recalls.begin(), recalls.end());
            size_t index_5 = static_cast<size_t>(recalls.size() * 0.05);
            size_t index_1 = static_cast<size_t>(recalls.size() * 0.01);
            p5_recall = recalls[index_5];
            p1_recall = recalls[index_1];
        }
    }

    std::sort(times.begin(), times.end());
    int64_t median_ms = times[times.size() / 2];
    return {median_ms, avg_recall, p5_recall, p1_recall, wae};
}

// ============================================================================
// main
// ============================================================================
int main() {
    const char* root_env = std::getenv("EXPERIMENTS_ROOT");
    if (!root_env) {
        std::cerr << "Error: EXPERIMENTS_ROOT is not set.\n"
                  << "  export EXPERIMENTS_ROOT=/path/to/experiments\n";
        return 1;
    }
    const std::filesystem::path root(root_env);
    std::cout << "EXPERIMENTS_ROOT: " << root << "\n\n";

    Eigen::setNbThreads(std::max(1u, std::thread::hardware_concurrency() / 4));

    // Fixed structural parameter
    constexpr size_t STATICS_LENGTH = 1 + 32 + 31 * 32; // 2-hop neighbors, M=16

    std::cout << "[GAMMA_SWEEP_HEADER] "
              << "dataset,gamma,avg_recall,p5_recall,p1_recall,median_time_ms,wae\n";

    for (const auto& ds : DATASETS) {
        std::cout << "\n======================================================\n"
                  << "Dataset: " << ds.dataset << "\n"
                  << "  k=" << ds.k
                  << "  alpha=" << ds.alpha
                  << "  expected_recall=" << ds.expected_recall
                  << "  ef_upper_bound=" << ds.ef_upper_bound
                  << "\n"
                  << "======================================================\n";

        // --- Load index + data (once per dataset) ---
        const std::string hdf5_path  = (root / "data"  / (ds.dataset + ".hdf5")).string();
        const std::string index_path = (root / "index" / (ds.dataset + "-M16-efc-500-parallel.hnsw")).string();

        auto [hnsw, query_ptr, data_ptr, gt_ptr, space] =
            load_index_and_data(hdf5_path, index_path, ds.metric);

        // --- Load samplings (pre-built, shared across all gamma values) ---
        const std::string samplings_path =
            (root / "sampling" / (ds.dataset + "-samplings--k" + std::to_string(ds.k) + "-ef.bin")).string();

        hnswdis::MatrixXf sample_q;
        hnswdis::MatrixXi sample_gt;
        hnswdis::MatrixXf sample_gt_dist;

        // Try loading with dist first; fall back to two-arg variant.
        try {
            hnswdis::deserialize_samplings(samplings_path, sample_q, sample_gt, sample_gt_dist);
        } catch (...) {
            hnswdis::deserialize_samplings(samplings_path, sample_q, sample_gt);
        }

        std::cout << "  Samplings loaded: " << sample_q.rows() << " queries\n";

        auto sample_q_ptr  = std::make_shared<hnswdis::MatrixXf>(sample_q);
        auto sample_gt_ptr = std::make_shared<hnswdis::MatrixXi>(sample_gt);

        // --- Sweep gamma ---
        for (float gamma : GAMMA_VALUES) {
            std::cout << "\n--- gamma=" << gamma << " ---\n";
            
            // Re-initialize the score calculator so it uses the swept gamma during the search!
            hnswdis::ApproximatedScoreCalculator score_cal(ds.alpha, gamma);

            // Build EfAdapter with this gamma (in-memory, not saved)
            hnswdis::EfAdapter ef_adapter(
                hnsw, data_ptr,
                ds.k, ds.metric,
                ds.expected_recall, ds.alpha, gamma,
                STATICS_LENGTH,
                sample_q_ptr, sample_gt_ptr,
                ds.ef_upper_bound);

            // Train CV buckets
            train_cv_buckets(
                ef_adapter, hnsw, data_ptr,
                ds.k, ds.metric,
                ds.alpha, gamma, STATICS_LENGTH,
                sample_q_ptr, sample_gt_ptr,
                ds.ef_upper_bound);

            const float wae = ef_adapter.get_wae();
            std::cout << "  WAE (weighted avg ef): " << static_cast<size_t>(wae) << "\n";

            hnswdis::Sketch sketch = make_sketch(ef_adapter, ds.expected_recall);

            auto [median_ms, avg_recall, p5_recall, p1_recall, _wae] = run_sweep(
                ds.dataset, ds.repeat,
                *hnsw, *query_ptr, *data_ptr, *gt_ptr,
                score_cal, ds.k, sketch,
                STATICS_LENGTH, ds.expected_recall, wae);

            std::cout << "  avg_recall=" << avg_recall
                      << "  p5_recall=" << p5_recall
                      << "  p1_recall=" << p1_recall
                      << "  median_time=" << median_ms << " ms\n";

            // Structured log line for the Python parser
            std::cout << "[GAMMA_SWEEP] "
                      << "dataset=" << ds.dataset
                      << " gamma=" << gamma
                      << " avg_recall=" << avg_recall
                      << " p5_recall=" << p5_recall
                      << " p1_recall=" << p1_recall
                      << " median_time_ms=" << median_ms
                      << " wae=" << static_cast<size_t>(wae)
                      << "\n";
        }
    }

    std::cout << "\nGamma sweep complete.\n";
    return 0;
}
