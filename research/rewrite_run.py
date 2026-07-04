import re

with open('experiments_driver/run.cpp', 'r') as f:
    content = f.read()

config_text = """
// ============================================================================
// GLOBAL CONFIGURATION
// Easily configure parameters here instead of modifying them in each function.
// ============================================================================
struct ExperimentConfig {
    std::string dataset;
    std::string metric;
    size_t k;
    float alpha;
    float gamma;
    float expected_recall;
    int ef_upper_bound;
    int repeat;
};

static std::vector<ExperimentConfig> g_experiments = {
    // dataset, metric, k, alpha, gamma, expected_recall, ef_upper_bound, repeat
    {"deep-image-96-angular", "cd", 100, 0.25f, 16.0f, 0.95f, 5000, 1},
    {"glove-100-angular", "cd", 100, 0.25f, 16.0f, 0.95f, 5000, 1},
    {"sift-128-euclidean", "l2", 100, 0.25f, 16.0f, 0.95f, 5000, 1},
    // {"msmarco", "cd", 1000, 0.25f, 16.0f, 0.95f, 5000, 1},
    // {"cohere", "cd", 1000, 0.25f, 16.0f, 0.95f, 5000, 1},
    // {"laion_image", "cd", 1000, 0.25f, 16.0f, 0.95f, 5000, 1},
    // {"laion_text", "cd", 1000, 0.25f, 16.0f, 0.95f, 5000, 1},
    // {"cluster_mg_uniform_100d", "cd", 1000, 0.251f, 16.0f, 0.95f, 5000, 1},
    // {"cluster_mg_zipf_100d", "cd", 1000, 0.25f, 16.0f, 0.95f, 5000, 1}
};
// ============================================================================
"""

# Insert it after #include <cstdlib>
content = re.sub(r'(#include <cstdlib>\n)', r'\1\n' + config_text + '\n', content)

# Now we need to replace the local configurations in each function.
# Specifically, we can search for `std::vector<std::tuple<...>> dataset_metrics = { ... };`
# and the loop `for (const auto ... : dataset_metrics)`
# and replace them with `for (const auto& conf : g_experiments)`

# Example: online_exp()
def replace_loop(func_name, code):
    # This regex is an approximation and might need manual tuning if it doesn't match
    pattern = r'(std::vector<std::(?:tuple|pair)[^;]+dataset_metrics\s*=\s*\{.*?\};)'
    match = re.search(pattern, code, re.DOTALL)
    if match:
        pass
    
    return code

# Since regex on C++ is hard, let's write a small script to just replace the blocks
