with open("hnswlib/hnswalg.h", "r") as f:
    lines = f.readlines()

def get_replacement1(indent="                "):
    return f"""{indent}bool is_visited = (visited_array[candidate_id] == visited_array_tag);
{indent}dist_t dist;
{indent}char *currObj1 = nullptr;
{indent}bool dist_computed = false;

{indent}if (flag_collect_statistics) {{
{indent}    currObj1 = (getDataByInternalId(candidate_id));
{indent}    dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
{indent}    dist_computed = true;
{indent}    edge_evals.push_back({{dist, is_visited}});

{indent}    if (edge_evals.size() == statics_limit) {{
{indent}        flag_collect_statistics = false;
{indent}        score = score_calculator.compute_score(data_point, *((size_t *) dist_func_param_), edge_evals.data(), edge_evals.size());
{indent}        if (sketch) {{
{indent}            ef = sketch->estimate_ef2(score); // used for estimating ef
{indent}            if (ef < ef_copy) {{
{indent}                ef = ef_copy;
{indent}            }}
{indent}        }} else {{
{indent}            ef = ef_copy;
{indent}        }}
{indent}        if (print_ef) std::cout << ef << ",";
{indent}        while (top_candidates.size() > ef) {{
{indent}            top_candidates.pop();
{indent}        }}
{indent}        if (!top_candidates.empty()) {{
{indent}            lowerBound = top_candidates.top().first;
{indent}        }}
{indent}    }}
{indent}}}

{indent}if (!is_visited) {{
{indent}    visited_array[candidate_id] = visited_array_tag;

{indent}    if (!dist_computed) {{
{indent}        currObj1 = (getDataByInternalId(candidate_id));
{indent}        dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
{indent}    }}

{indent}    bool flag_consider_candidate;
{indent}    if (!bare_bone_search && stop_condition) {{
{indent}        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
{indent}    }} else {{
{indent}        if (dist_computed && edge_evals.size() <= statics_limit) {{
{indent}            flag_consider_candidate = true;
{indent}        }} else {{
{indent}            flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
{indent}        }}
{indent}    }}

{indent}    if (flag_consider_candidate) {{
{indent}        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
{indent}        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ + offsetLevel0_, _MM_HINT_T0);
#endif

{indent}        if (bare_bone_search || (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {{
{indent}            top_candidates.emplace(dist, candidate_id);
{indent}            if (!bare_bone_search && stop_condition) {{
{indent}                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
{indent}            }}
{indent}        }}

{indent}        bool flag_remove_extra = false;
{indent}        if (!bare_bone_search && stop_condition) {{
{indent}            flag_remove_extra = stop_condition->should_remove_extra();
{indent}        }} else {{
{indent}            flag_remove_extra = top_candidates.size() > ef;
{indent}        }}
{indent}        while (flag_remove_extra) {{
{indent}            tableint id = top_candidates.top().second;
{indent}            top_candidates.pop();
{indent}            if (!bare_bone_search && stop_condition) {{
{indent}                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
{indent}                flag_remove_extra = stop_condition->should_remove_extra();
{indent}            }} else {{
{indent}                flag_remove_extra = top_candidates.size() > ef;
{indent}            }}
{indent}        }}

{indent}        if (!top_candidates.empty())
{indent}            lowerBound = top_candidates.top().first;
{indent}    }}
{indent}}}\n"""

def get_replacement2(indent="                "):
    return f"""{indent}bool is_visited = (visited_array[candidate_id] == visited_array_tag);
{indent}dist_t dist;
{indent}char *currObj1 = nullptr;
{indent}bool dist_computed = false;

{indent}if (flag_collect_statistics) {{
{indent}    currObj1 = (getDataByInternalId(candidate_id));
{indent}    dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
{indent}    dist_computed = true;
{indent}    edge_evals.push_back({{dist, is_visited}});

{indent}    if (edge_evals.size() == statics_limit) {{
{indent}        flag_collect_statistics = false;
{indent}        score = score_calculator.compute_score(data_point, *((size_t *) dist_func_param_), edge_evals.data(), edge_evals.size());
{indent}        if (sketch) {{
{indent}            ef = sketch->estimate_ef2(score);  // get estimated ef
{indent}            if (ef < ef_copy) {{
{indent}                ef = ef_copy;
{indent}            }}
{indent}        }} else {{
{indent}            ef = ef_copy;
{indent}        }}
{indent}        while (top_candidates.size() > ef) {{
{indent}            top_candidates.pop();
{indent}        }}
{indent}        if (!top_candidates.empty()) {{
{indent}            lowerBound = top_candidates.top().first;
{indent}        }}
{indent}    }}
{indent}}}

{indent}if (!is_visited) {{
{indent}    visited_array[candidate_id] = visited_array_tag;

{indent}    if (!dist_computed) {{
{indent}        currObj1 = (getDataByInternalId(candidate_id));
{indent}        dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
{indent}    }}

{indent}    bool flag_consider_candidate;
{indent}    if (!bare_bone_search && stop_condition) {{
{indent}        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
{indent}    }} else {{
{indent}        if (dist_computed && edge_evals.size() <= statics_limit) {{
{indent}            flag_consider_candidate = true;
{indent}        }} else {{
{indent}            flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
{indent}        }}
{indent}    }}

{indent}    if (flag_consider_candidate) {{
{indent}        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
{indent}        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ + offsetLevel0_, _MM_HINT_T0);
#endif

{indent}        if (bare_bone_search || (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {{
{indent}            top_candidates.emplace(dist, candidate_id);
{indent}            if (!bare_bone_search && stop_condition) {{
{indent}                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
{indent}            }}
{indent}        }}

{indent}        bool flag_remove_extra = false;
{indent}        if (!bare_bone_search && stop_condition) {{
{indent}            flag_remove_extra = stop_condition->should_remove_extra();
{indent}        }} else {{
{indent}            flag_remove_extra = top_candidates.size() > ef;
{indent}        }}
{indent}        while (flag_remove_extra) {{
{indent}            tableint id = top_candidates.top().second;
{indent}            top_candidates.pop();
{indent}            if (!bare_bone_search && stop_condition) {{
{indent}                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
{indent}                flag_remove_extra = stop_condition->should_remove_extra();
{indent}            }} else {{
{indent}                flag_remove_extra = top_candidates.size() > ef;
{indent}            }}
{indent}        }}

{indent}        if (!top_candidates.empty())
{indent}            lowerBound = top_candidates.top().first;
{indent}    }}
{indent}}}\n"""

# Block 2: lines 1765-1851 (0-indexed: 1764 to 1851)
lines[1764:1851] = [get_replacement2()]

# Replace variables in block 2
for i in range(1675, 1715):
    if "dist_t distances[statics_limit];" in lines[i]:
        lines[i] = "        std::vector<std::pair<dist_t, bool>> edge_evals;\n"
    elif "size_t size_distances = 0;" in lines[i]:
        lines[i] = "        edge_evals.reserve(statics_limit);\n"
    elif "distances[size_distances] = dist;" in lines[i]:
        lines[i] = "                edge_evals.push_back({dist, false});\n"
    elif "++size_distances;" in lines[i]:
        lines[i] = ""

# Block 1: lines 1499-1589 (0-indexed: 1498 to 1589)
lines[1498:1589] = [get_replacement1()]

# Replace variables in block 1
for i in range(1410, 1450):
    if "dist_t distances[statics_limit];" in lines[i]:
        lines[i] = "        std::vector<std::pair<dist_t, bool>> edge_evals;\n"
    elif "size_t size_distances = 0;" in lines[i]:
        lines[i] = "        edge_evals.reserve(statics_limit);\n"
    elif "distances[size_distances] = dist;" in lines[i]:
        lines[i] = "                edge_evals.push_back({dist, false});\n"
    elif "++size_distances;" in lines[i]:
        lines[i] = ""

with open("hnswlib/hnswalg.h", "w") as f:
    f.writelines(lines)

