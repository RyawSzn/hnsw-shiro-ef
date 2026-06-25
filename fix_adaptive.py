import re

with open("hnswlib/hnswalg.h", "r") as f:
    text = f.read()

# 1. Change distances to edge_evals inside adaptive functions
pattern1 = r'''size_t ef_copy = ef;
\s*ef = std::numeric_limits<size_t>::max\(\);
\s*bool flag_collect_statistics = true;
\s*size_t statics_limit = statics_length;
\s*dist_t distances\[statics_limit\];
\s*size_t size_distances = 0;'''

replacement1 = '''size_t ef_copy = ef;
        ef = std::numeric_limits<size_t>::max();
        bool flag_collect_statistics = true;
        size_t statics_limit = statics_length;
        std::vector<std::pair<dist_t, bool>> edge_evals;
        edge_evals.reserve(statics_limit);'''

text = re.sub(pattern1, replacement1, text)

# 2. Change entry point tracking
pattern2 = r'''top_candidates\.emplace\(dist, ep_id\);
\s*// change === start
\s*if \(flag_collect_statistics\)\{
\s*distances\[size_distances\] = dist;
\s*\+\+size_distances;
\s*\}'''

replacement2 = '''top_candidates.emplace(dist, ep_id);
            // change === start
            if (flag_collect_statistics){
                edge_evals.push_back({dist, false});
            }'''

text = re.sub(pattern2, replacement2, text)

# 3. Change loop in adaptiveSearchBaseLayerST

loop_pattern1 = r'''\s*if \(!\(visited_array\[candidate_id\] == visited_array_tag\)\) \{
\s*visited_array\[candidate_id\] = visited_array_tag;
\s*char \*currObj1 = \(getDataByInternalId\(candidate_id\)\);
\s*dist_t dist = fstdistfunc_\(data_point, currObj1, dist_func_param_\);
\s*bool flag_consider_candidate;
\s*if \(!bare_bone_search && stop_condition\) \{
\s*flag_consider_candidate = stop_condition->should_consider_candidate\(dist, lowerBound\);
\s*\} else \{
\s*// change === start
\s*if \(flag_collect_statistics\)\{
\s*flag_consider_candidate = true;
\s*if \(edge_evals\.size\(\) == statics_limit\)\{ // let the search run for statics_limit iterations
\s*flag_collect_statistics = false;
\s*score = score_calculator\.compute_score\(data_point, \*\(\(size_t \*\) dist_func_param_\), edge_evals\.data\(\), edge_evals\.size\(\)\);
\s*if \(sketch\)\{
\s*ef = sketch->estimate_ef2\(score\); // used for estimating ef
\s*if \(ef < ef_copy\)\{
\s*ef = ef_copy;
\s*\}
\s*\} else \{
\s*ef = ef_copy;
\s*\}
\s*// print ef: for ef distribution analysis
\s*if \(print_ef\)\{
\s*std::cout << ef << ",";
\s*\}
\s*// shrinking top_candidates to ef in case it is larger
\s*while\(top_candidates\.size\(\) > ef\)\{
\s*top_candidates\.pop\(\);
\s*\}
\s*if \(!top_candidates\.empty\(\)\)\{
\s*lowerBound = top_candidates\.top\(\)\.first;
\s*\}
\s*\}
\s*\} else \{
\s*// change === end
\s*flag_consider_candidate = top_candidates\.size\(\) < ef \|\| lowerBound > dist;
\s*\}
\s*\}
\s*if \(flag_consider_candidate\) \{
\s*candidate_set\.emplace\(-dist, candidate_id\);
#ifdef USE_SSE
\s*_mm_prefetch\(data_level0_memory_ \+ candidate_set\.top\(\)\.second \* size_data_per_element_ \+
\s*offsetLevel0_,  ///////////
\s*_MM_HINT_T0\);  ////////////////////////
#endif
\s*if \(bare_bone_search \|\|
\s*\(!isMarkedDeleted\(candidate_id\) && \(\(!isIdAllowed\) \|\| \(\*isIdAllowed\)\(getExternalLabel\(candidate_id\)\)\)\)\) \{
\s*top_candidates\.emplace\(dist, candidate_id\);
\s*// change === start
\s*if \(flag_collect_statistics\)\{
\s*edge_evals\.push_back\(\{dist, false\}\);
\s*\}
\s*// change === end
\s*if \(!bare_bone_search && stop_condition\) \{
\s*stop_condition->add_point_to_result\(getExternalLabel\(candidate_id\), currObj1, dist\);
\s*\}
\s*\}
\s*bool flag_remove_extra = false;
\s*if \(!bare_bone_search && stop_condition\) \{
\s*flag_remove_extra = stop_condition->should_remove_extra\(\);
\s*\} else \{
\s*flag_remove_extra = top_candidates\.size\(\) > ef;
\s*\}
\s*while \(flag_remove_extra\) \{
\s*tableint id = top_candidates\.top\(\)\.second;
\s*top_candidates\.pop\(\);
\s*if \(!bare_bone_search && stop_condition\) \{
\s*stop_condition->remove_point_from_result\(getExternalLabel\(id\), getDataByInternalId\(id\), dist\);
\s*flag_remove_extra = stop_condition->should_remove_extra\(\);
\s*\} else \{
\s*flag_remove_extra = top_candidates\.size\(\) > ef;
\s*\}
\s*\}
\s*if \(!top_candidates\.empty\(\)\)
\s*lowerBound = top_candidates\.top\(\)\.first;
\s*\}
\s*\}'''

loop_replacement1 = '''bool is_visited = (visited_array[candidate_id] == visited_array_tag);
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
                            ef = sketch->estimate_ef2(score); // used for estimating ef
                            if (ef < ef_copy){
                                ef = ef_copy;
                            }
                        } else {
                            ef = ef_copy;
                        }
                        if (print_ef) std::cout << ef << ",";
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
                }'''

# Replace adaptiveSearchBaseLayerST loop
text = re.sub(loop_pattern1, loop_replacement1, text, count=1)

# Now adaptiveSearchBaseLayerST2 (slightly different sketch comment: " // get estimated ef")
loop_pattern2 = loop_pattern1.replace('// used for estimating ef', '// get estimated ef').replace('if (print_ef){\nstd::cout << ef << ",";\n}\n', '').replace('if (print_ef){\n\s*std::cout << ef << ",";\n\s*}', '')

# We can reuse the same replacement logic but strip print_ef
loop_replacement2 = loop_replacement1.replace('if (print_ef) std::cout << ef << ",";', '')

text = re.sub(loop_pattern2, loop_replacement2, text, count=1)

with open("hnswlib/hnswalg.h", "w") as f:
    f.write(text)

