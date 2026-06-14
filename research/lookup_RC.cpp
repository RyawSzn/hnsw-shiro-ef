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
#include "../hnswlib/distribution.h"

using namespace hnswlib;

int main(int argc, char** argv) {
    float target_recall = 0.95f;
    std::string dataset = "glove-100-angular";

    if (argc > 1) target_recall = std::stof(argv[1]);
    if (argc > 2) dataset = argv[2];
    
    std::cout << "=== Generating 20x20 Adaptive EF Lookup Table (M_RC x m_LID) ===" << std::endl;
    std::cout << "Target Recall: " << target_recall << std::endl;
    std::cout << "Dataset:       " << dataset << std::endl;
    std::cout << "----------------------------------------------------------------\n" << std::endl;

    const char *experiments_root = std::getenv("EXPERIMENTS_ROOT");
    std::filesystem::path root = experiments_root ? std::filesystem::path(experiments_root)
                                                  : std::filesystem::current_path();
    std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
    if (!std::filesystem::exists(hdf5_path)) {
        hdf5_path = (root / "experiments/data" / (dataset + ".hdf5")).string();
    }

    std::cout << "Loading dataset: " << hdf5_path << std::endl;
    hnswdis::MatrixXf full_data;
    hnswdis::MatrixXf query_vectors;
    hnswdis::MatrixXi ground_truth;

    load_hdf5(hdf5_path, query_vectors, full_data, ground_truth);
    normalize_matrix(full_data);
    normalize_matrix(query_vectors);

    std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();
    std::cout << "Loading HNSW index from " << index_path << "..." << std::endl;
    L2Space space(full_data.cols());
    HierarchicalNSW<float>* alg_hnsw = new HierarchicalNSW<float>(&space, index_path, false, full_data.rows());

    int nq = query_vectors.rows();
    std::vector<float> M_scores(nq, 0.0f);
    std::vector<float> m_scores(nq, 0.0f);

    hnswdis::CosineSimilarityEstimator estimator(full_data);

    std::cout << "\nPhase 1: Probing all queries to calculate M_RC and m_LID..." << std::endl;
    alg_hnsw->setEf(50); // baseline probe ef
    int K_probe = 32;

    #pragma omp parallel for
    for (int i = 0; i < nq; i++) {
        Eigen::RowVectorXf q = query_vectors.row(i);
        const float* query = q.data();
        
        auto [th_mean, th_var] = estimator.get_practical_distribution(q);
        float d_mean = std::max(0.01f, 1.0f - (float)th_mean);
        
        tableint currObj = alg_hnsw->enterpoint_node_;
        float curdist = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(currObj), alg_hnsw->dist_func_param_);
        
        int L = alg_hnsw->maxlevel_;
        
        // Fast-forward through upper layers without tracking E_l
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
        
        // Base layer (0) Probe
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
        float b_0 = base_dists[0];
        
        // Compute M_RC (Relative Contrast)
        M_scores[i] = b_0 / d_mean;
        
        // Compute m_LID
        int K = std::min(K_probe, (int)base_dists.size());
        float m_q = 0;
        if (K > 1) {
            float d_K = base_dists[K-1] + 1e-6f;
            float sum_log = 0;
            for (int k = 0; k < K - 1; k++) {
                float d_i = base_dists[k] + 1e-6f;
                sum_log += std::log(d_K / d_i);
            }
            if (sum_log > 0) m_q = (K - 1) / sum_log;
        }
        m_scores[i] = m_q;
    }

    std::cout << "Phase 2: Calculating 20x20 Quantile Boundaries..." << std::endl;
    std::vector<float> sorted_M = M_scores;
    std::vector<float> sorted_m = m_scores;
    std::sort(sorted_M.begin(), sorted_M.end());
    std::sort(sorted_m.begin(), sorted_m.end());

    std::vector<float> M_bounds(21);
    std::vector<float> m_bounds(21);
    for(int i=0; i<=20; i++) {
        int idx = std::min((int)(nq - 1), (int)(i * nq / 20.0));
        M_bounds[i] = sorted_M[idx];
        m_bounds[i] = sorted_m[idx];
    }
    M_bounds[20] = std::numeric_limits<float>::max();
    m_bounds[20] = std::numeric_limits<float>::max();

    std::vector<int> buckets[20][20];
    for (int i = 0; i < nq; i++) {
        int bM = 0;
        while(bM < 19 && M_scores[i] > M_bounds[bM+1]) bM++;
        int bm = 0;
        while(bm < 19 && m_scores[i] > m_bounds[bm+1]) bm++;
        buckets[bM][bm].push_back(i);
    }

    std::cout << "Phase 3: Stepping ef by +50 to hit Target Recall (" << target_recall << ") for each bucket...\n" << std::endl;

    std::vector<std::vector<std::pair<int, float>>> lookup_table(20, std::vector<std::pair<int, float>>(20, {0, 0.0f}));

    // Output Table Formatting
    std::cout << "=======================================================================================================================" << std::endl;
    std::cout << "                                  20x20 Adaptive EF Lookup Table (ef, actual_recall)                                   " << std::endl;
    std::cout << "=======================================================================================================================" << std::endl;
    
    // Print Header
    std::cout << "     |";
    for(int j=0; j<20; j++) std::cout << "    m" << std::setw(2) << std::left << (j+1) << "    |";
    std::cout << "\n-----------------------------------------------------------------------------------------------------------------------" << std::endl;

    std::ofstream out("research/lookup_table_RC.csv");
    out << "M_bin,m_bin,ef,actual_recall\n";

    for (int i = 0; i < 20; i++) {
        std::cout << " M" << std::setw(2) << std::left << (i+1) << " |";
        for (int j = 0; j < 20; j++) {
            auto& query_list = buckets[i][j];
            if (query_list.empty()) {
                std::cout << "   ---     |";
                continue;
            }

            int best_ef = 50;
            float best_recall = 0.0f;

            for (int ef = 50; ef <= 2000; ef += 50) {
                alg_hnsw->setEf(ef);
                int total_hits = 0;

                for (int q_idx : query_list) {
                    auto res = alg_hnsw->searchKnn(query_vectors.row(q_idx).data(), 10);
                    while (!res.empty()) {
                        tableint id = res.top().second;
                        res.pop();
                        for (int c = 0; c < 10; c++) {
                            if (ground_truth(q_idx, c) == id) {
                                total_hits++;
                                break;
                            }
                        }
                    }
                }

                float avg_recall = (float)total_hits / (query_list.size() * 10);
                best_ef = ef;
                best_recall = avg_recall;

                if (avg_recall >= target_recall) break; 
            }

            lookup_table[i][j] = {best_ef, best_recall};
            out << (i+1) << "," << (j+1) << "," << best_ef << "," << best_recall << "\n";
            out.flush();
            
            char buf[32];
            snprintf(buf, sizeof(buf), "%3d,%.2f", best_ef, best_recall);
            std::cout << " " << std::setw(8) << std::left << buf << "|";
        }
        std::cout << "\n";
    }
    std::cout << "=======================================================================================================================\n" << std::endl;

    out.close();
    std::cout << "Lookup table saved to research/lookup_table_RC.csv" << std::endl;

    delete alg_hnsw;
    return 0;
}
