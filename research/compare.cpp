#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <queue>
#include <unordered_set>
#include <chrono>
#include <algorithm>
#include <iomanip>

#include "../experiments_driver/util.h"
#include "../hnswlib/hnswlib.h"

using namespace hnswlib;

int main(int argc, char** argv) {
    std::string dataset = "glove-100-angular";
    if (argc > 1) dataset = argv[1];

    const char *experiments_root = std::getenv("EXPERIMENTS_ROOT");
    std::filesystem::path root = experiments_root ? std::filesystem::path(experiments_root)
                                                  : std::filesystem::current_path();
    std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
    if (!std::filesystem::exists(hdf5_path)) {
        hdf5_path = (root / "experiments/data" / (dataset + ".hdf5")).string();
    }

    hnswdis::MatrixXf full_data;
    hnswdis::MatrixXf query_vectors;
    hnswdis::MatrixXi ground_truth;

    load_hdf5(hdf5_path, query_vectors, full_data, ground_truth);
    normalize_matrix(full_data);
    normalize_matrix(query_vectors);

    std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();
    L2Space space(full_data.cols());
    HierarchicalNSW<float>* alg_hnsw = new HierarchicalNSW<float>(&space, index_path, false, full_data.rows());

    int nq = query_vectors.rows();
    alg_hnsw->setEf(50); // Use baseline ef=50
    int K_probe = 32;

    std::ofstream out("research/compare_metrics.csv");
    out << "query_id,recall,m_LID,e_eps01,e_eps02,e_eps05\n";

    std::cout << "Probing all queries to compare LID vs Eps-Hardness..." << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < nq; i++) {
        const float* query = query_vectors.row(i).data();
        
        tableint currObj = alg_hnsw->enterpoint_node_;
        float curdist = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(currObj), alg_hnsw->dist_func_param_);
        
        int L = alg_hnsw->maxlevel_;
        for (int level = L; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                auto* data = reinterpret_cast<unsigned int*>(alg_hnsw->get_linklist(currObj, level));
                int sz = alg_hnsw->getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
                auto* nbrs = reinterpret_cast<tableint*>(data + 1);
                for (int j = 0; j < sz; j++) {
                    tableint cand = nbrs[j];
                    float d = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(cand), alg_hnsw->dist_func_param_);
                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
        
        std::priority_queue<std::pair<float, tableint>> top_candidates;
        std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>, std::greater<std::pair<float, tableint>>> candidate_set;
        std::unordered_set<tableint> visited;
        
        visited.insert(currObj);
        candidate_set.emplace(curdist, currObj);
        top_candidates.emplace(curdist, currObj);
        
        std::vector<float> base_dists;
        base_dists.push_back(curdist);
        float lowerBound = curdist;
        
        while (!candidate_set.empty()) {
            auto current_node_pair = candidate_set.top();
            if (current_node_pair.first > lowerBound && top_candidates.size() == 50) break;
            candidate_set.pop();
            tableint cur_node = current_node_pair.second;
            
            auto* data = reinterpret_cast<int*>(alg_hnsw->get_linklist0(cur_node));
            int sz = alg_hnsw->getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
            auto* nbrs = reinterpret_cast<tableint*>(data + 1);
            
            for (int j = 0; j < sz; j++) {
                tableint cand = nbrs[j];
                if (visited.find(cand) == visited.end()) {
                    visited.insert(cand);
                    float d = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(cand), alg_hnsw->dist_func_param_);
                    base_dists.push_back(d);
                    if (top_candidates.size() < 50 || lowerBound > d) {
                        candidate_set.emplace(d, cand);
                        top_candidates.emplace(d, cand);
                        if (top_candidates.size() > 50) top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }
            }
        }
        
        std::sort(base_dists.begin(), base_dists.end());
        int K = std::min(K_probe, (int)base_dists.size());
        
        // 1. Compute m_LID
        float m_LID = 0;
        if (K > 1) {
            float d_K = base_dists[K-1] + 1e-6f;
            float sum_log = 0;
            for (int k = 0; k < K - 1; k++) {
                float d_i = base_dists[k] + 1e-6f;
                sum_log += std::log(d_K / d_i);
            }
            if (sum_log > 0) m_LID = (K - 1) / sum_log;
        }

        // 2. Compute e_eps (Probe-local eps-hardness)
        // We evaluate how many points are within (1+eps) of the nearest neighbor d_1
        int e_01 = 0, e_02 = 0, e_05 = 0;
        if (K > 0) {
            float d_1 = base_dists[0] + 1e-6f;
            for (int k = 0; k < K; k++) {
                if (base_dists[k] <= (1.0f + 0.1f) * d_1) e_01++;
                if (base_dists[k] <= (1.0f + 0.2f) * d_1) e_02++;
                if (base_dists[k] <= (1.0f + 0.5f) * d_1) e_05++;
            }
        }

        // Compute actual recall
        auto res = alg_hnsw->searchKnn(query, 10);
        int hits = 0;
        while (!res.empty()) {
            tableint id = res.top().second;
            res.pop();
            for (int c = 0; c < 10; c++) {
                if (ground_truth(i, c) == id) {
                    hits++;
                    break;
                }
            }
        }
        float recall = hits / 10.0f;
        
        #pragma omp critical
        {
            out << i << "," << recall << "," << m_LID << "," << e_01 << "," << e_02 << "," << e_05 << "\n";
        }
    }

    out.close();
    std::cout << "Done! Metrics saved to research/compare_metrics.csv" << std::endl;

    delete alg_hnsw;
    return 0;
}
