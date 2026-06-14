import re

with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/research/lookup.cpp', 'r') as f:
    code = f.read()

old_w = """    // Hardcoded optimal data-driven absolute correlation weights for GloVe L=5
    // Extracted from our previous experiment (Log Calculation)
    std::vector<float> w(L + 1, 0.0f);
    if (L >= 5) {
        w[5] = 0.0000f;
        w[4] = 0.1617f;
        w[3] = 0.2761f;
        w[2] = 0.3176f;
        w[1] = 0.2446f;
    } else {
        w[1] = 1.0f; // Fallback
    }"""

new_w = """    // Exponential weight: w_l = e^{-l + L + 1} / sum_{j=0}^L e^j
    std::vector<float> w(L + 1, 0.0f);
    float denom = 0.0f;
    for (int j = 0; j <= L; j++) {
        denom += std::exp((float)j);
    }
    std::cout << "\\nWeight Distribution (Exponential):" << std::endl;
    for (int l = 0; l <= L; l++) {
        w[l] = std::exp(-l + L + 1.0f) / denom;
        std::cout << "  w[" << l << "] = " << w[l] << std::endl;
    }"""

if old_w in code:
    code = code.replace(old_w, new_w)
    # Ensure M loop includes l=0 so the biggest weight is used?
    # Wait, if they excluded 1 before, let's check the loop.
    old_loop = "for (int l = L; l >= 1; l--) {"
    new_loop = "for (int l = L; l >= 1; l--) { // Note: l=0 is excluded as per E_L..1 constraint"
    code = code.replace(old_loop, new_loop)
    
    with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/research/lookup.cpp', 'w') as f:
        f.write(code)
    print("Patched lookup.cpp with exponential weights.")
else:
    print("Could not find weight block.")
