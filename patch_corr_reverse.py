with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/research/corr.cpp', 'r') as f:
    code = f.read()

old_E = """        // Compute E_l as Wasted Effort (1.0 - eta_l)
        std::vector<float> E(L + 1, 0.0f);
        for (int l = L; l >= 1; l--) {
            float eta_l = (n_evals[l] > 0) ? ((float)n_improves[l] / n_evals[l]) : 0.0f;
            E[l] = (n_evals[l] > 0) ? (1.0f - eta_l) : 0.0f;
        }"""

new_E = """        // Compute E_l (Original Progress Ratio)
        std::vector<float> E(L + 1, 0.0f);
        float prev_b = dist_before_layer;
        for (int l = L; l >= 1; l--) {
            float rho_l = (prev_b - b_best[l]) / (prev_b + 1e-6f);
            E[l] = std::log1p(n_evals[l]) / (1.0f + std::max(0.0f, rho_l));
            prev_b = b_best[l];
        }"""

if old_E in code:
    code = code.replace(old_E, new_E)
    with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/research/corr.cpp', 'w') as f:
        f.write(code)
    print("Reverted E_l calculation to original.")
else:
    print("Could not find the target code to replace.")
