// gamma_buckets.cpp
// Exports the dynamic EfAdapter lookup table (Score -> EF mapping) for each gamma.
// Outputs a CSV to stdout: dataset,gamma,score,ef

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

int main() {
    const char* root_env = std::getenv("EXPERIMENTS_ROOT");
    if (!root_env) {
        std::cerr << "Error: EXPERIMENTS_ROOT is not set.\n";
        return 1;
    }
    const std::filesystem::path root(root_env);
    
    std::cerr << "EXPERIMENTS_ROOT: " << root << "\n\n";
    Eigen::setNbThreads(std::max(1u, std::thread::hardware_concurrency() / 4));

    constexpr size_t STATICS_LENGTH = 1 + 32 + 31 * 32;

    std::cout << "dataset,gamma,score,ef\n";

    for (const auto& ds : DATASETS) {
        std::cerr << "Processing Dataset: " << ds.dataset << "\n";

        const std::string hdf5_path  = (root / "data"  / (ds.dataset + ".hdf5")).string();
        const std::string index_path = (root / "index" / (ds.dataset + "-M16-efc-500-parallel.hnsw")).string();

        auto [hnsw, query_ptr, data_ptr, gt_ptr, space] =
            load_index_and_data(hdf5_path, index_path, ds.metric);

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
        std::cerr << "  Samplings loaded.\n";

        auto sample_q_ptr  = std::make_shared<hnswdis::MatrixXf>(sample_q);
        auto sample_gt_ptr = std::make_shared<hnswdis::MatrixXi>(sample_gt);

        for (float gamma : GAMMA_VALUES) {
            std::cerr << "  Sweep gamma=" << gamma << "\n";
            
            hnswdis::EfAdapter ef_adapter(
                hnsw, data_ptr, ds.k, ds.metric, ds.expected_recall, ds.alpha, gamma,
                STATICS_LENGTH, sample_q_ptr, sample_gt_ptr, ds.ef_upper_bound);

            ef_adapter.init_with_cv_buckets(
                hnsw, data_ptr, ds.k, ds.metric, ds.alpha, gamma, STATICS_LENGTH,
                sample_q_ptr, sample_gt_ptr, N_DEP_TABLES);

            hnswdis::Sketch sketch = make_sketch(ef_adapter, ds.expected_recall);
            
            // We use a median CV score (e.g. 20.0) to query the lookup table.
            float cv_median = 20.0f;
            
            // Map every Score (0 to 100) to an EF
            for (int score = 0; score <= 100; ++score) {
                size_t mapped_ef = sketch.estimate_ef2(static_cast<float>(score), cv_median);
                std::cout << ds.dataset << "," 
                          << gamma << "," 
                          << score << "," 
                          << mapped_ef << "\n";
            }
        }
    }

    std::cerr << "Bucket mapping extraction complete.\n";
    return 0;
}
