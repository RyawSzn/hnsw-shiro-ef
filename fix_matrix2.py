import re

with open('hnswlib/adaptive_ef.h', 'r') as f:
    code = f.read()

# I will find the block starting with `int actual_n_cv_tables = std::max(1, n_cv_tables);`
m = re.search(r'(int actual_n_cv_tables = std::max\(1, n_cv_tables\);.*?wae = accumulated_wae;)', code, flags=re.DOTALL)
if m:
    old_body = m.group(1)
    new_body = """float accumulated_wae = 0.0f;

            if (n_cv_tables == 0) {
                std::map<int, std::vector<int>> cv_groups;
                for (int i = 0; i < n; ++i) {
                    int cv_score = std::max(0, std::min(100, static_cast<int>(cvs[i] * 100.0f)));
                    cv_groups[cv_score].push_back(i);
                }

                for (auto it = cv_groups.begin(); it != cv_groups.end(); ) {
                    if (it->second.size() < static_cast<size_t>(std::max(1, min_queries_per_score))) {
                        it = cv_groups.erase(it);
                    } else {
                        ++it;
                    }
                }

                cv_tables.resize(cv_groups.size());
                cv_centers.clear();

                int t = 0;
                for (const auto &[cv_score, q_indices] : cv_groups) {
                    cv_centers.push_back(cv_score / 100.0f);
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
                int actual_n_cv_tables = std::max(1, n_cv_tables);
                int chunk = n / actual_n_cv_tables;

                cv_centers.clear();
                for (int t = 0; t < actual_n_cv_tables; ++t) {
                    int lo = t * chunk;
                    int hi = (t == actual_n_cv_tables - 1) ? n : (t + 1) * chunk;
                    cv_centers.push_back(cvs[order[(lo + hi) / 2]]);
                }

                cv_tables.resize(actual_n_cv_tables);

                for (int t = 0; t < actual_n_cv_tables; ++t)
                {
                    int lo = t * chunk;
                    int hi = (t == actual_n_cv_tables - 1) ? n : (t + 1) * chunk;

                    int bucket_size = hi - lo;
                    MatrixXf bucket_queries(bucket_size, query_vectors->cols());
                    MatrixXi bucket_gt(bucket_size, ground_truth_ptr->cols());
                    for (int r = 0; r < bucket_size; ++r)
                    {
                        bucket_queries.row(r) = query_vectors->row(order[lo + r]);
                        bucket_gt.row(r)      = ground_truth_ptr->row(order[lo + r]);
                    }

                    std::cout << "Training rv-bucket " << t
                              << " [" << cvs[order[lo]] << ", "
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
            }

            wae = accumulated_wae;"""

    code = code[:m.start(1)] + new_body + code[m.end(1):]

    with open('hnswlib/adaptive_ef.h', 'w') as f:
        f.write(code)

