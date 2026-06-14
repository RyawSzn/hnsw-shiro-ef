#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <queue>
#include <unordered_set>
#include <chrono>

#include "../experiments_driver/util.h"
#include "../hnswlib/hnswlib.h"

using namespace hnswlib;

int main(int argc, char** argv) {
    int ef = 50;
    std::string dataset = "glove-100-angular";

    if (argc > 1) ef = std::stoi(argv[1]);
    if (argc > 2) dataset = argv[2];
    
    std::cout << "--- Data-Driven Weight Extraction ---" << std::endl;
    std::cout << "ef:            " << ef << std::endl;
    std::cout << "Dataset:       " << dataset << std::endl;
    std::cout << "-------------------------------------\n" << std::endl;

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

    alg_hnsw->setEf(ef);
    
    int nq = query_vectors.rows();
    int L = alg_hnsw->maxlevel_;
    int K_probe = 32;
    
    std::ofstream out("research/raw_E_results.csv");
    out << "query_id,m,recall";
    for(int l=L; l>=1; l--) out << ",E" << l;
    out << "\n";

    std::cout << "Running queries and extracting E_l..." << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < nq; i++) {
        const float* query = query_vectors.row(i).data();
        
        tableint currObj = alg_hnsw->enterpoint_node_;
        float curdist = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(currObj), alg_hnsw->dist_func_param_);
        
        std::vector<int> n_evals(L + 1, 0);
        std::vector<int> n_improves(L + 1, 0);
        std::vector<float> b_best(L + 1, 0.0f);
        
        float dist_before_layer = curdist;
        
        // Upper layers (L down to 1)
        for (int level = L; level > 0; level--) {
            int evals = 0;
            int improves = 0;
            bool changed = true;
            while (changed) {
                changed = false;
                auto* data = reinterpret_cast<unsigned int*>(alg_hnsw->get_linklist(currObj, level));
                int sz = alg_hnsw->getListCount(reinterpret_cast<hnswlib::linklistsizeint*>(data));
                auto* nbrs = reinterpret_cast<tableint*>(data + 1);
                
                for (int j = 0; j < sz; j++) {
                    tableint cand = nbrs[j];
                    float d = alg_hnsw->fstdistfunc_(query, alg_hnsw->getDataByInternalId(cand), alg_hnsw->dist_func_param_);
                    evals++;
                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                        improves++;
                    }
                }
            }
            n_evals[level] = evals;
            n_improves[level] = improves;
            b_best[level] = curdist;
        }
        
        // Base layer (0)
        int evals_0 = 0;
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
            if (current_node_pair.first > lowerBound && top_candidates.size() == ef) break;
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
                    evals_0++;
                    base_dists.push_back(d);
                    
                    if (top_candidates.size() < ef || lowerBound > d) {
                        candidate_set.emplace(d, cand);
                        top_candidates.emplace(d, cand);
                        if (top_candidates.size() > ef) top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }
            }
        }
        n_evals[0] = evals_0;
        
        if (top_candidates.size() > 0) { 
            curdist = *std::min_element(base_dists.begin(), base_dists.end());
        }
        b_best[0] = curdist;
        
        // Compute E_l (Original Progress Ratio with log)
        std::vector<float> E(L + 1, 0.0f);
        float prev_b = dist_before_layer;
        for (int l = L; l >= 1; l--) {
            float rho_l = (prev_b - b_best[l]) / (prev_b + 1e-6f);
            E[l] = std::log1p(n_evals[l]) / (1.0f + std::max(0.0f, rho_l));
            prev_b = b_best[l];
        }
        
        // Compute m(q)
        std::sort(base_dists.begin(), base_dists.end());
        int K = std::min(K_probe, (int)base_dists.size());
        float m_q = 0;
        if (K > 1) {
            float d_K = base_dists[K-1] + 1e-6f;
            float sum_log = 0;
            for (int k = 0; k < K - 1; k++) {
                float d_i = base_dists[k] + 1e-6f;
                sum_log += std::log(d_K / d_i);
            }
            if (sum_log > 0) {
                m_q = (K - 1) / sum_log;
            } else {
                m_q = 0;
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
            out << i << "," << m_q << "," << recall;
            for(int l=L; l>=1; l--) out << "," << E[l];
            out << "\n";
        }
    }
    
    out.close();
    std::cout << "Done! Raw E_l saved to research/raw_E_results.csv" << std::endl;
    delete alg_hnsw;
    return 0;
}
