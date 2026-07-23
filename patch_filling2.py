import sys

with open('./hnswlib/shiro_ef.h', 'r') as f:
    content = f.read()

old_str = """            else {
                // Implement a different smoothing strategy if needed
            }"""

new_str = """            else if constexpr (FILLING_METHOD == 2) {
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

                std::vector<float> efs(101, 0.0f);
                
                // 2. IDW (Inverse Distance Weighting) for all scores 0-100
                for (int s = 0; s <= 100; ++s) {
                    if (score_to_ef.count(s)) {
                        efs[s] = score_to_ef[s];
                    } else {
                        float sum_w = 0.0f;
                        float sum_ef = 0.0f;
                        for (auto const& [known_s, known_ef] : score_to_ef) {
                            float dist = std::abs(s - known_s);
                            float w = 1.0f / (dist * dist);
                            sum_w += w;
                            sum_ef += w * known_ef;
                        }
                        if (sum_w > 0) {
                            efs[s] = sum_ef / sum_w;
                        } else {
                            efs[s] = 0.0f;
                        }
                    }
                }

                // 3. 1D PAVA (Pool-Adjacent-Violators Algorithm) to enforce monotonicity (decreasing)
                std::vector<std::pair<float, int>> blocks;
                for (int i = 0; i <= 100; ++i) {
                    float val = efs[i];
                    int weight = 1;
                    
                    while (!blocks.empty() && blocks.back().first < val) 
                    {
                        float prev_val = blocks.back().first;
                        int prev_weight = blocks.back().second;
                        
                        val = (prev_val * prev_weight + val * weight) / (prev_weight + weight);
                        weight += prev_weight;
                        
                        blocks.pop_back();
                    }
                    blocks.push_back({val, weight});
                }
                
                int idx = 0;
                for (const auto& block : blocks) {
                    for (int i = 0; i < block.second; ++i) {
                        efs[idx++] = block.first;
                    }
                }

                // 4. Build smoothed table and calculate Weighted Average EF
                EfRecallTable smoothed_table;
                float weighted_average_ef = 0.0f;
                float total_queries = query_vectors->rows();

                for (int s = 0; s <= 100; ++s) {
                    size_t final_ef = std::round(efs[s]);
                    smoothed_table.push_back({s, {{(int)final_ef, expected_recall}}});

                    if (score_to_cnt.count(s)) {
                        weighted_average_ef += score_to_cnt[s] * final_ef / total_queries;
                    }
                }

                out_table = smoothed_table;
                std::cout << "Weighted average ef: " << weighted_average_ef << std::endl;
                wae = weighted_average_ef;
            }"""

if old_str in content:
    with open('./hnswlib/shiro_ef.h', 'w') as f:
        f.write(content.replace(old_str, new_str))
    print("Success")
else:
    print("Failed to find old string")
