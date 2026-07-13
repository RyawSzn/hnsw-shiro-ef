with open('hnswlib/adaptive_ef.h', 'r') as f:
    code = f.read()

old_init = """            hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);
            std::vector<float> cvs = collect_cv(*alg_hnsw, *query_vectors, score_cal, k, statics_length);

            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return cvs[a] < cvs[b]; });

            int actual_n_cv_tables = std::max(1, n_cv_tables);
            int chunk = n / actual_n_cv_tables;

            cv_centers.resize(actual_n_cv_tables);
            for (int t = 0; t < actual_n_cv_tables; ++t) {
                int lo = t * chunk;
                int hi = (t == actual_n_cv_tables - 1) ? n : (t + 1) * chunk;
                float sum = 0;
                for (int i = lo; i < hi; ++i) sum += cvs[order[i]];
                cv_centers[t] = sum / (hi - lo);
            }

            cv_tables.resize(actual_n_cv_tables);
            float accumulated_wae = 0.0f;

            for (int t = 0; t < actual_n_cv_tables; ++t)
            {
                int lo = t * chunk;
                int hi = (t == actual_n_cv_tables - 1) ? n : (t + 1) * chunk;
                int bucket_size = hi - lo;

                MatrixXf bucket_queries(bucket_size, query_vectors->cols());
                MatrixXi bucket_gt(bucket_size, ground_truth_ptr->cols());

                for (int i = 0; i < bucket_size; ++i) {
                    bucket_queries.row(i) = query_vectors->row(order[lo + i]);
                    bucket_gt.row(i) = ground_truth_ptr->row(order[lo + i]);
                }

                std::cout << "Training cv-bucket " << t << " [" 
                          << (t > 0 ? cvs[order[lo]] : 0.0f) << ", " 
                          << (t < actual_n_cv_tables - 1 ? cvs[order[hi]] : std::numeric_limits<float>::infinity())
                          << ") with " << bucket_size << " queries." << std::endl;

                init(alg_hnsw,
                     data_vectors,
                     k, metric, alpha, gamma, statics_length,
                     std::make_shared<MatrixXf>(bucket_queries),
                     std::make_shared<MatrixXi>(bucket_gt),
                     cv_tables[t], min_queries_per_score);

                accumulated_wae += wae * ((float)bucket_size / n);
            }

            wae = accumulated_wae;
        }"""

new_init = """            hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);
            std::vector<float> cvs = collect_cv(*alg_hnsw, *query_vectors, score_cal, k, statics_length);

            float accumulated_wae = 0.0f;

            if (n_cv_tables == 0) {
                // Remove percentile bucketing method. 
                // Group queries by exact floor(CV * 100) and construct a 100x100 matrix lookup directly.
                std::map<int, std::vector<int>> cv_groups;
                for (int i = 0; i < n; ++i) {
                    int cv_score = std::max(0, std::min(100, static_cast<int>(cvs[i] * 100.0f)));
                    cv_groups[cv_score].push_back(i);
                }

                // Filter out groups with insufficient queries to train a stable table
                for (auto it = cv_groups.begin(); it != cv_groups.end(); ) {
                    if (it->second.size() < static_cast<size_t>(std::max(1, min_queries_per_score))) {
                        it = cv_groups.erase(it);
                    } else {
                        ++it;
                    }
                }

                cv_tables.resize(cv_groups.size());
                cv_centers.resize(cv_groups.size());

                int t = 0;
                for (const auto &[cv_score, q_indices] : cv_groups) {
                    cv_centers[t] = cv_score / 100.0f;
                    int bucket_size = q_indices.size();

                    MatrixXf bucket_queries(bucket_size, query_vectors->cols());
                    MatrixXi bucket_gt(bucket_size, ground_truth_ptr->cols());

                    for (int i = 0; i < bucket_size; ++i) {
                        bucket_queries.row(i) = query_vectors->row(q_indices[i]);
                        bucket_gt.row(i) = ground_truth_ptr->row(q_indices[i]);
                    }

                    std::cout << "Training absolute cv-matrix bin " << cv_score << " with " << bucket_size << " queries." << std::endl;

                    init(alg_hnsw, data_vectors, k, metric, alpha, gamma, statics_length,
                         std::make_shared<MatrixXf>(bucket_queries),
                         std::make_shared<MatrixXi>(bucket_gt),
                         cv_tables[t], min_queries_per_score);

                    accumulated_wae += wae * ((float)bucket_size / n);
                    t++;
                }
            } else {
                // Standard percentile-based bucketing
                std::vector<int> order(n);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int a, int b) { return cvs[a] < cvs[b]; });

                int actual_n_cv_tables = std::max(1, n_cv_tables);
                int chunk = n / actual_n_cv_tables;

                cv_centers.resize(actual_n_cv_tables);
                for (int t = 0; t < actual_n_cv_tables; ++t) {
                    int lo = t * chunk;
                    int hi = (t == actual_n_cv_tables - 1) ? n : (t + 1) * chunk;
                    float sum = 0;
                    for (int i = lo; i < hi; ++i) sum += cvs[order[i]];
                    cv_centers[t] = sum / (hi - lo);
                }

                cv_tables.resize(actual_n_cv_tables);

                for (int t = 0; t < actual_n_cv_tables; ++t)
                {
                    int lo = t * chunk;
                    int hi = (t == actual_n_cv_tables - 1) ? n : (t + 1) * chunk;
                    int bucket_size = hi - lo;

                    MatrixXf bucket_queries(bucket_size, query_vectors->cols());
                    MatrixXi bucket_gt(bucket_size, ground_truth_ptr->cols());

                    for (int i = 0; i < bucket_size; ++i) {
                        bucket_queries.row(i) = query_vectors->row(order[lo + i]);
                        bucket_gt.row(i) = ground_truth_ptr->row(order[lo + i]);
                    }

                    std::cout << "Training cv-bucket " << t << " [" 
                              << (t > 0 ? cvs[order[lo]] : 0.0f) << ", " 
                              << (t < actual_n_cv_tables - 1 ? cvs[order[hi]] : std::numeric_limits<float>::infinity())
                              << ") with " << bucket_size << " queries." << std::endl;

                    init(alg_hnsw,
                         data_vectors,
                         k, metric, alpha, gamma, statics_length,
                         std::make_shared<MatrixXf>(bucket_queries),
                         std::make_shared<MatrixXi>(bucket_gt),
                         cv_tables[t],
                         min_queries_per_score);

                    accumulated_wae += wae * ((float)bucket_size / n);
                }
            }

            wae = accumulated_wae;
        }"""

code = code.replace(old_init, new_init)

with open('hnswlib/adaptive_ef.h', 'w') as f:
    f.write(code)

