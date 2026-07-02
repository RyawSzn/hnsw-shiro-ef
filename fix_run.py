import re

with open('experiments_driver/run.cpp', 'r') as f:
    content = f.read()

old_block = """    std::vector<std::vector<size_t>> result;
    std::vector<float> scores;
    for (const auto &r : search_score_result) {
        result.push_back(std::move(r.first));
        scores.push_back(r.second);
    }"""

new_block = """    std::vector<std::vector<size_t>> result;
    std::vector<float> scores;
    std::vector<float> cvs;
    for (const auto &r : search_score_result) {
        result.push_back(std::move(std::get<0>(r)));
        scores.push_back(std::get<1>(r));
        cvs.push_back(std::get<2>(r));
    }"""

content = content.replace(old_block, new_block)

old_write1 = """    out << "query_id,score,recall\\n";
    for (size_t i = 0; i < recalls.size(); ++i) {
        out << i << "," << scores[i] << "," << recalls[i] << "\\n";
    }"""

new_write1 = """    out << "query_id,score,cv,recall\\n";
    for (size_t i = 0; i < recalls.size(); ++i) {
        out << i << "," << scores[i] << "," << cvs[i] << "," << recalls[i] << "\\n";
    }"""

content = content.replace(old_write1, new_write1)

with open('experiments_driver/run.cpp', 'w') as f:
    f.write(content)
