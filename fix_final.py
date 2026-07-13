import re

with open('experiments_driver/run.cpp', 'r') as f:
    code = f.read()

# Fix online_exp baseline_search
code = re.sub(r'baseline_search\(dataset, repeat, \*hnsw, \*query, \*ground_truth, k, ef_upper_bound, sampling_size\);',
              r'baseline_search(dataset, repeat, *hnsw, *query, *ground_truth, k, ef_upper_bound);', code)
              
# Fix sensitivity_analysis baseline_search
code = re.sub(r'baseline_search\(dataset, repeat, \*hnsw, \*query, \*ground_truth, k, ef_upper_bound, sampling_size\);',
              r'baseline_search(dataset, repeat, *hnsw, *query, *ground_truth, k, ef_upper_bound);', code)

# Fix insert_exp_adaef_update EfAdapter constructor
code = re.sub(r'(alg_hnsw, full_data_ptr, k, metric, expected_recall, alpha, gamma, statics_length, sample_query_vectors_ptr, sample_ground_truth_ptr, ef_upper_bound), sampling_size\)',
              r'\1)', code)

# Fix delete_exp_setup EfAdapter constructor
code = re.sub(r'(alg_hnsw, after_updates_data_ptr, k, metric, expected_recall, alpha, gamma, statics_length, sample_query_vectors, sample_ground_truth_ptr, ef_upper_bound), sampling_size\)',
              r'\1)', code)

# Fix delete_exp_adaef_update EfAdapter constructor
code = re.sub(r'(alg_hnsw, after_updates_data_ptr, k, metric, expected_recall, alpha, gamma, statics_length, sample_query_vectors_ptr, sample_ground_truth_ptr, ef_upper_bound), sampling_size\)',
              r'\1)', code)

with open('experiments_driver/run.cpp', 'w') as f:
    f.write(code)

