import re

with open("hnswlib/hnswalg.h", "r") as f:
    content = f.read()

def replace_block(text):
    # Find the start of the block
    pattern = r'(#endif\s*)if \(!\(visited_array\[candidate_id\] == visited_array_tag\)\) \{'
    
    matches = list(re.finditer(pattern, text))
    print(f"Found {len(matches)} matches")
    
    if len(matches) == 0:
        return text

    new_text = text
    # we replace from the end to not mess up indices
    for m in reversed(matches):
        start_idx = m.start(0)
        
        # find matching closing brace
        idx = start_idx + len(m.group(1)) + len("if (!(visited_array[candidate_id] == visited_array_tag)) {") - 1
        brace_count = 1
        idx += 1
        while brace_count > 0 and idx < len(new_text):
            if new_text[idx] == '{':
                brace_count += 1
            elif new_text[idx] == '}':
                brace_count -= 1
            idx += 1
            
        end_idx = idx
        
        # Replacement code
        replacement = m.group(1) + """bool is_visited = (visited_array[candidate_id] == visited_array_tag);
                dist_t dist;
                char *currObj1 = nullptr;
                bool dist_computed = false;

                if (flag_collect_statistics) {
                    currObj1 = (getDataByInternalId(candidate_id));
                    dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    dist_computed = true;
                    edge_evals.push_back({dist, is_visited});

                    if (edge_evals.size() == statics_limit) {
                        flag_collect_statistics = false;
                        score = score_calculator.compute_score(data_point, *((size_t *) dist_func_param_), edge_evals.data(), edge_evals.size());
                        if (sketch){
                            ef = sketch->estimate_ef2(score);
                            if (ef < ef_copy){
                                ef = ef_copy;
                            }
                        } else {
                            ef = ef_copy;
                        }

                        while(top_candidates.size() > ef){
                            top_candidates.pop();
                        }
                        if (!top_candidates.empty()){
                            lowerBound = top_candidates.top().first;
                        }
                    }
                }

                if (!is_visited) {
                    visited_array[candidate_id] = visited_array_tag;

                    if (!dist_computed) {
                        currObj1 = (getDataByInternalId(candidate_id));
                        dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    }

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        if (dist_computed && edge_evals.size() <= statics_limit) {
                            flag_consider_candidate = true;
                        } else {
                            flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                        }
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ + offsetLevel0_, _MM_HINT_T0);
#endif

                        if (bare_bone_search || (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }"""
        new_text = new_text[:start_idx] + replacement + new_text[end_idx:]
    return new_text

new_content = replace_block(content)

with open("hnswlib/hnswalg.h", "w") as f:
    f.write(new_content)

