import re

with open('hnswlib/adaptive_ef.h', 'r') as f:
    content = f.read()

old_logic = """            auto &stat = first_recall_estimator.get_recall_statistics();
            float weighted_average_ef = 0.0f;
            for (int i = 0; i < (int)stat.size(); ++i)
            {
                size_t cnt = std::get<5>(stat[i]);
                size_t ef = out_table[i].second.back().first;
                for (size_t j = 0; j < out_table[i].second.size() - 1; ++j)
                {
                    if (out_table[i].second[j].second >= expected_recall)
                    {
                        ef = out_table[i].second[j].first;
                        break;
                    }
                }

                weighted_average_ef += cnt * ef / (float)query_vectors->rows();
            }
            std::cout << "Weighted average ef: " << weighted_average_ef << std::endl;
            wae = weighted_average_ef;
        }"""

new_logic = """            auto &stat = first_recall_estimator.get_recall_statistics();

            // 1. Extract score to EF and count mapping
            std::map<int, size_t> score_to_ef;
            std::map<int, size_t> score_to_cnt;

            for (int i = 0; i < (int)stat.size(); ++i)
            {
                int score = std::get<0>(stat[i]);
                size_t cnt = std::get<5>(stat[i]);
                size_t ef = out_table[i].second.back().first;
                for (size_t j = 0; j < out_table[i].second.size() - 1; ++j)
                {
                    if (out_table[i].second[j].second >= expected_recall)
                    {
                        ef = out_table[i].second[j].first;
                        break;
                    }
                }
                score_to_ef[score] = ef;
                score_to_cnt[score] = cnt;
            }

            // 2. Identify continuous and discrete parts
            std::vector<int> scores;
            for (auto const& [score, _] : score_to_ef) {
                scores.push_back(score);
            }

            int best_start = 0, best_len = 0;
            int curr_start = 0, curr_len = 1;

            if (scores.size() > 0) {
                for (size_t i = 1; i < scores.size(); ++i) {
                    if (scores[i] == scores[i-1] + 1) {
                        curr_len++;
                    } else {
                        if (curr_len > best_len) {
                            best_len = curr_len;
                            best_start = curr_start;
                        }
                        curr_start = i;
                        curr_len = 1;
                    }
                }
                if (curr_len > best_len) {
                    best_len = curr_len;
                    best_start = curr_start;
                }
            }

            int cont_start_score = 0;
            int cont_end_score = 100;
            if (best_len > 0) {
                cont_start_score = scores[best_start];
                cont_end_score = scores[best_start + best_len - 1];
                if (best_len >= 3) {
                    cont_start_score += 1;
                    cont_end_score -= 1;
                }
            }

            // 3. Compute weighted averages for discrete regions
            float left_sum_ef = 0, right_sum_ef = 0;
            size_t left_sum_cnt = 0, right_sum_cnt = 0;

            for (auto const& [score, ef] : score_to_ef) {
                size_t cnt = score_to_cnt[score];
                if (score < cont_start_score) {
                    left_sum_ef += ef * cnt;
                    left_sum_cnt += cnt;
                } else if (score > cont_end_score) {
                    right_sum_ef += ef * cnt;
                    right_sum_cnt += cnt;
                }
            }

            size_t left_avg_ef = left_sum_cnt > 0 ? std::round(left_sum_ef / left_sum_cnt) : (score_to_ef.empty() ? 0 : score_to_ef.begin()->second);
            size_t right_avg_ef = right_sum_cnt > 0 ? std::round(right_sum_ef / right_sum_cnt) : (score_to_ef.empty() ? 0 : score_to_ef.rbegin()->second);

            // 4. Build smoothed table and calculate new weighted average EF
            EfRecallTable smoothed_table;
            float weighted_average_ef = 0.0f;
            float total_queries = query_vectors->rows();

            for (int s = 0; s <= 100; ++s) {
                size_t final_ef = 0;
                size_t cnt = score_to_cnt.count(s) ? score_to_cnt[s] : 0;

                if (s < cont_start_score) {
                    final_ef = left_avg_ef;
                    smoothed_table.push_back({s, {{(int)final_ef, expected_recall}}});
                } else if (s > cont_end_score) {
                    final_ef = right_avg_ef;
                    smoothed_table.push_back({s, {{(int)final_ef, expected_recall}}});
                } else {
                    if (score_to_ef.count(s)) {
                        final_ef = score_to_ef[s];
                        smoothed_table.push_back({s, {{(int)final_ef, expected_recall}}});
                    }
                }

                if (cnt > 0 && final_ef > 0) {
                    weighted_average_ef += cnt * final_ef / total_queries;
                }
            }

            out_table = smoothed_table;
            std::cout << "Weighted average ef: " << weighted_average_ef << std::endl;
            wae = weighted_average_ef;
        }"""

content = content.replace(old_logic, new_logic)

with open('hnswlib/adaptive_ef.h', 'w') as f:
    f.write(content)
