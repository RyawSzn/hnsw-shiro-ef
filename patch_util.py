import re

with open('/home/ryawszn/dev/cpp/hnsw-shiro-ef/experiments_driver/util.h', 'r') as f:
    content = f.read()

# Add efs to IterationResult
content = re.sub(
    r'std::vector<float> recalls;\n\s*};',
    r'std::vector<float> recalls;\n        std::vector<size_t> efs;\n    };',
    content
)

# Add std::vector<size_t> efs(num_queries);
content = re.sub(
    r'std::vector<float> recalls\(num_queries\);',
    r'std::vector<float> recalls(num_queries);\n        std::vector<size_t> efs(num_queries);',
    content
)

# Call adaptiveSearchKnnTest with out_ef
content = re.sub(
    r'auto pq = alg_hnsw\.adaptiveSearchKnnTest\(\n\s*query_vectors\.row\(j\)\.data\(\), k, statics_length, score_cal, &sketch\);',
    r'size_t query_ef = ef;\n            auto pq = alg_hnsw.adaptiveSearchKnnTest(\n                query_vectors.row(j).data(), k, statics_length, score_cal, &sketch, nullptr, &query_ef);\n            efs[j] = query_ef;',
    content
)

# Update iter_results[rep] assignment
content = re.sub(
    r'iter_results\[rep\] = \{avg_latency, latencies_ns, recalls\};',
    r'iter_results[rep] = {avg_latency, latencies_ns, recalls, efs};',
    content
)

# In the attempt_file write
content = re.sub(
    r'attempt_file << j << "," << ef << "," << iter_results\[rep\]\.latencies_ns\[j\]',
    r'attempt_file << j << "," << iter_results[rep].efs[j] << "," << iter_results[rep].latencies_ns[j]',
    content
)

# In median_iter allocation
content = re.sub(
    r'median_iter\.recalls\.resize\(num_queries\);',
    r'median_iter.recalls.resize(num_queries);\n    median_iter.efs.resize(num_queries);',
    content
)

# In median_iter assignment
content = re.sub(
    r'median_iter\.recalls\[j\] = iter_results\[0\]\.recalls\[j\];',
    r'median_iter.recalls[j] = iter_results[0].recalls[j];\n        median_iter.efs[j] = iter_results[0].efs[j];',
    content
)

# In median_iter write
content = re.sub(
    r'csv_file << j << "," << ef << "," << median_iter\.latencies_ns\[j\]',
    r'csv_file << j << "," << median_iter.efs[j] << "," << median_iter.latencies_ns[j]',
    content
)

with open('/home/ryawszn/dev/cpp/hnsw-shiro-ef/experiments_driver/util.h', 'w') as f:
    f.write(content)

