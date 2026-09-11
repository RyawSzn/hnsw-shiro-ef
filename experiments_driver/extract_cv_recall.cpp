#include "util.h"
#include "../hnswlib/shiro_ef.h"
#include <filesystem>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_name>\n";
        return 1;
    }
    std::string dataset = argv[1];
    
    char *root_path = std::getenv("EXPERIMENTS_ROOT");
    if (!root_path) {
        std::cerr << "Set EXPERIMENTS_ROOT environment variable (e.g. /home/ryawszn/experiments/2metric)\n";
        return 1;
    }
    std::filesystem::path root(root_path);

    std::string metric = "cd"; // fallback
    if (dataset.find("sift") != std::string::npos || dataset.find("gist") != std::string::npos || dataset.find("tiny") != std::string::npos) {
        metric = "l2";
    }

    std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
    std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();
    
    std::cout << "Loading dataset: " << dataset << "\n";
    std::cout << "HDF5: " << hdf5_path << "\n";
    std::cout << "Index: " << index_path << "\n";

    auto tuple = load_index_and_data(hdf5_path, index_path, metric);
    auto hnsw = std::get<0>(tuple);
    auto query = std::get<1>(tuple);
    auto ground_truth = std::get<3>(tuple);

    size_t k = 100;
    size_t stats_length = 1 + 32 + 31 * 32; // 2-hop budget probe (1025)
    float alpha = 1.0f;
    float gamma = 12.0f;
    hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);

    std::filesystem::create_directories("research/csv");
    std::string out_path = "research/csv/cv_recall_" + dataset + ".csv";
    std::ofstream out(out_path);
    out << "query_id,cv,recall\n";

    hnsw->setEf(100);

    for (int j = 0; j < query->rows(); ++j) {
        float cv = 0.0f;
        
        auto ret_cv = hnsw->adaptiveSearchKnn(query->row(j).data(), k, stats_length, score_cal, nullptr, &cv);
        
        auto ret = hnsw->searchKnn(query->row(j).data(), k);
        size_t count = ret.size();
        std::vector<size_t> labels(count);
        while (!ret.empty()) {
            labels[--count] = ret.top().second;
            ret.pop();
        }

        int correct = 0;
        for (const auto& item : labels) {
            for (int gt_idx = 0; gt_idx < (int)k; ++gt_idx) {
                if (item == (*ground_truth)(j, gt_idx)) {
                    correct++;
                    break;
                }
            }
        }
        float recall = static_cast<float>(correct) / k;
        out << j << "," << cv << "," << recall << "\n";
        
        if (j % 1000 == 0) {
            std::cout << "Processed " << j << "/" << query->rows() << " queries\r" << std::flush;
        }
    }
    std::cout << "\n";
    out.close();
    std::cout << "Done writing to " << out_path << std::endl;
    return 0;
}
