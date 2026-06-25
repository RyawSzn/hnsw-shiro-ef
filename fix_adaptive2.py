import re

with open("hnswlib/hnswalg.h", "r") as f:
    lines = f.readlines()

def replace_block(lines, func_name):
    # Find function
    func_idx = -1
    for i, line in enumerate(lines):
        if func_name + "(" in line:
            func_idx = i
            break
            
    if func_idx == -1: return lines
    
    # Find loop start inside function
    start_idx = -1
    for i in range(func_idx, len(lines)):
        if "if (!(visited_array[candidate_id] == visited_array_tag)) {" in lines[i]:
            start_idx = i
            break
            
    if start_idx == -1: return lines
    
    # Find matching brace
    brace_count = 0
    end_idx = -1
    for i in range(start_idx, len(lines)):
        brace_count += lines[i].count("{")
        brace_count -= lines[i].count("}")
        if brace_count == 0:
            end_idx = i
            break

    indent = "                "
    replacement = [
        f"{indent}bool is_visited = (visited_array[candidate_id] == visited_array_tag);\n",
        f"{indent}dist_t dist;\n",
        f"{indent}char *currObj1 = nullptr;\n",
        f"{indent}bool dist_computed = false;\n",
        f"\n",
        f"{indent}if (flag_collect_statistics) {{\n",
        f"{indent}    currObj1 = (getDataByInternalId(candidate_id));\n",
        f"{indent}    dist = fstdistfunc_(data_point, currObj1, dist_func_param_);\n",
        f"{indent}    dist_computed = true;\n",
        f"{indent}    edge_evals.push_back({{dist, is_visited}});\n",
        f"\n",
        f"{indent}    if (edge_evals.size() == statics_limit) {{\n",
        f"{indent}        flag_collect_statistics = false;\n",
        f"{indent}        score = score_calculator.compute_score(data_point, *((size_t *) dist_func_param_), edge_evals.data(), edge_evals.size());\n",
        f"{indent}        if (sketch) {{\n",
        f"{indent}            ef = sketch->estimate_ef2(score);\n",
        f"{indent}            if (ef < ef_copy) {{\n",
        f"{indent}                ef = ef_copy;\n",
        f"{indent}            }}\n",
        f"{indent}        }} else {{\n",
        f"{indent}            ef = ef_copy;\n",
        f"{indent}        }}\n",
        f"\n",
        f"{indent}        while (top_candidates.size() > ef) {{\n",
        f"{indent}            top_candidates.pop();\n",
        f"{indent}        }}\n",
        f"{indent}        if (!top_candidates.empty()) {{\n",
        f"{indent}            lowerBound = top_candidates.top().first;\n",
        f"{indent}        }}\n",
        f"{indent}    }}\n",
        f"{indent}}}\n",
        f"\n",
        f"{indent}if (!is_visited) {{\n",
        f"{indent}    visited_array[candidate_id] = visited_array_tag;\n",
        f"\n",
        f"{indent}    if (!dist_computed) {{\n",
        f"{indent}        currObj1 = (getDataByInternalId(candidate_id));\n",
        f"{indent}        dist = fstdistfunc_(data_point, currObj1, dist_func_param_);\n",
        f"{indent}    }}\n",
        f"\n",
        f"{indent}    bool flag_consider_candidate;\n",
        f"{indent}    if (!bare_bone_search && stop_condition) {{\n",
        f"{indent}        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);\n",
        f"{indent}    }} else {{\n",
        f"{indent}        if (dist_computed && edge_evals.size() <= statics_limit) {{\n",
        f"{indent}            flag_consider_candidate = true;\n",
        f"{indent}        }} else {{\n",
        f"{indent}            flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;\n",
        f"{indent}        }}\n",
        f"{indent}    }}\n",
        f"\n",
        f"{indent}    if (flag_consider_candidate) {{\n",
        f"{indent}        candidate_set.emplace(-dist, candidate_id);\n",
        "#ifdef USE_SSE\n",
        f"{indent}        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ + offsetLevel0_, _MM_HINT_T0);\n",
        "#endif\n",
        f"\n",
        f"{indent}        if (bare_bone_search || (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {{\n",
        f"{indent}            top_candidates.emplace(dist, candidate_id);\n",
        f"{indent}            if (!bare_bone_search && stop_condition) {{\n",
        f"{indent}                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);\n",
        f"{indent}            }}\n",
        f"{indent}        }}\n",
        f"\n",
        f"{indent}        bool flag_remove_extra = false;\n",
        f"{indent}        if (!bare_bone_search && stop_condition) {{\n",
        f"{indent}            flag_remove_extra = stop_condition->should_remove_extra();\n",
        f"{indent}        }} else {{\n",
        f"{indent}            flag_remove_extra = top_candidates.size() > ef;\n",
        f"{indent}        }}\n",
        f"{indent}        while (flag_remove_extra) {{\n",
        f"{indent}            tableint id = top_candidates.top().second;\n",
        f"{indent}            top_candidates.pop();\n",
        f"{indent}            if (!bare_bone_search && stop_condition) {{\n",
        f"{indent}                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);\n",
        f"{indent}                flag_remove_extra = stop_condition->should_remove_extra();\n",
        f"{indent}            }} else {{\n",
        f"{indent}                flag_remove_extra = top_candidates.size() > ef;\n",
        f"{indent}            }}\n",
        f"{indent}        }}\n",
        f"\n",
        f"{indent}        if (!top_candidates.empty())\n",
        f"{indent}            lowerBound = top_candidates.top().first;\n",
        f"{indent}    }}\n",
        f"{indent}}}\n"
    ]
    
    return lines[:start_idx] + replacement + lines[end_idx+1:]

lines = replace_block(lines, "adaptiveSearchBaseLayerST")
lines = replace_block(lines, "adaptiveSearchBaseLayerSTWithPatienceInProximity")

with open("hnswlib/hnswalg.h", "w") as f:
    f.writelines(lines)

