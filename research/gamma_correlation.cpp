// gamma_correlation.cpp
// Evaluates the correlation between Routing Value (score) and Actual Recall
// for different values of gamma in the full dynamic lookup system.
// Outputs a CSV to stdout: dataset,gamma,query_id,score,recall

#include "../experiments_driver/util.h"
#include "../hnswlib/adaptive_ef.h"
#include <filesystem>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

static const std::vector<float> GAMMA_VALUES = {
    1.0f, 2.0f, 4.0f, 8.0f, 12.0f, 16.0f, 24.0f, 32.0f, 48.0f, 64.0f
};

struct DatasetConfig {
    std::string dataset;
    std::string metric;
    size_t      k;
    float       alpha;
    float       expected_recall;
    int         ef_upper_bound;
};

static const std::vector<DatasetConfig> DATASETS = {
    {"deep-image-96-angular", "cd",  100, 0.25f, 0.95f, 4000},
    {"glove-100-angular",     "cd",  100, 0.25f, 0.95f, 4000},
    {"sift-128-euclidean",    "l2",  100, 0.25f, 0.95f,  350},
};

static constexpr int N_DEP_TABLES = 10;

static hnswdis::Sketch make_sketch(const hnswdis::EfAdapter& adapter, float expected_recall) {
    if (adapter.has_cv_tables())
        return hnswdis::Sketch(adapter.get_all_tables(), adapter.get_cv_centers(), expected_recall);
    return hnswdis::Sketch(adapter.get_ef_recall_estimators(), expected_recall);
}

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

int main() {
    const char* root_env = std::getenv("EXPERIMENTS_ROOT");
    if (!root_env) {
        std::cerr << "Error: EXPERIMENTS_ROOT is not set.\n"
                  << "  export EXPERIMENTS_ROOT=/path/to/experiments\n";
        return 1;
    }
    const std::filesystem::path root(root_env);
    
    std::cerr << "EXPERIMENTS_ROOT: " << root << "\n\n";

    Eigen::setNbThreads(std::max(1u, std::thread::hardware_concurrency() / 4));

    constexpr size_t STATICS_LENGTH = 1 + 32 + 31 * 32;

    // CSV Header
    std::cout << "dataset,gamma,query_id,score,recall\n";

    for (const auto& ds : DATASETS) {
        std::cerr << "Processing Dataset: " << ds.dataset << "\n";

        const std::string hdf5_path  = (root / "data"  / (ds.dataset + ".hdf5")).string();
        const std::string index_path = (root / "index" / (ds.dataset + "-M16-efc-500-parallel.hnsw")).string();

        auto [hnsw, query_ptr, data_ptr, gt_ptr, space] =
            load_index_and_data(hdf5_path, index_path, ds.metric);

        // Load samplings to build the lookup table
        const std::string samplings_path =
            (root / "sampling" / (ds.dataset + "-samplings--k" + std::to_string(ds.k) + "-ef.bin")).string();

        hnswdis::MatrixXf sample_q;
        hnswdis::MatrixXi sample_gt;
        hnswdis::MatrixXf sample_gt_dist;

        try {
            hnswdis::deserialize_samplings(samplings_path, sample_q, sample_gt, sample_gt_dist);
        } catch (...) {
            hnswdis::deserialize_samplings(samplings_path, sample_q, sample_gt);
        }
        std::cerr << "  Samplings loaded: " << sample_q.rows() << " queries\n";

        auto sample_q_ptr  = std::make_shared<hnswdis::MatrixXf>(sample_q);
        auto sample_gt_ptr = std::make_shared<hnswdis::MatrixXi>(sample_gt);

        for (float gamma : GAMMA_VALUES) {
            std::cerr << "  Sweep gamma=" << gamma << "\n";
            
            // 1. Initialize score calculator with swept gamma
            hnswdis::ApproximatedScoreCalculator score_cal(ds.alpha, gamma);
            
            // 2. We run the search_score_and_cv just to extract the pure RV scores
            auto search_score_result = hnswdis::hnsw_search_score_and_cv(
                *hnsw, *query_ptr, *data_ptr, score_cal, ds.k, STATICS_LENGTH);

            std::vector<float> scores;
            scores.reserve(search_score_result.size());
            for (const auto &r : search_score_result) {
                scores.push_back(std::get<1>(r));
            }

            // 3. Build EfAdapter (lookup table) with this gamma
            hnswdis::EfAdapter ef_adapter(
                hnsw, data_ptr, ds.k, ds.metric, ds.expected_recall, ds.alpha, gamma,
                STATICS_LENGTH, sample_q_ptr, sample_gt_ptr, ds.ef_upper_bound);

            train_cv_buckets(
                ef_adapter, hnsw, data_ptr, ds.k, ds.metric, ds.alpha, gamma, STATICS_LENGTH,
                sample_q_ptr, sample_gt_ptr, ds.ef_upper_bound);

            float wae = ef_adapter.get_wae();
            hnswdis::Sketch sketch = make_sketch(ef_adapter, ds.expected_recall);
            
            // Set base ef to the WAE
            hnsw->setEf(static_cast<size_t>(wae));

            // 4. Run the full dynamic adaptive search using the sketch lookup
            std::vector<std::vector<size_t>> result_labels(query_ptr->rows(), std::vector<size_t>(ds.k, 0));
            
            for (int j = 0; j < query_ptr->rows(); ++j) {
                auto pq = hnsw->adaptiveSearchKnnTest(
                    query_ptr->row(j).data(), ds.k, STATICS_LENGTH, score_cal, &sketch);
                size_t cnt = pq.size();
                while (!pq.empty()) {
                    result_labels[j][--cnt] = pq.top().second;
                    pq.pop();
                }
            }

            // 5. Compute actual recall under the dynamic system
            auto recalls = hnswdis::compute_recall(*gt_ptr, result_labels, ds.k, false);

            for (size_t i = 0; i < recalls.size(); ++i) {
                std::cout << ds.dataset << "," 
                          << gamma << "," 
                          << i << "," 
                          << scores[i] << "," 
                          << recalls[i] << "\n";
            }
        }
    }

    std::cerr << "Gamma correlation sweep complete.\n";
    return 0;
}
