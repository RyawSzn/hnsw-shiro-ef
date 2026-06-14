with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/research/lookup_5x10.cpp', 'r') as f:
    code = f.read()

# fix the broken cout
code = code.replace('std::cout << "Phase 3: Stepping ef by +50 for each regime...\\n"; out.flush(); << std::endl;', 'std::cout << "Phase 3: Stepping ef by +50 for each regime...\\n" << std::endl;')

# add flush
old_out = 'out << (i+1) << "," << (j+1) << "," << best_ef << "," << best_recall << "\\n";'
new_out = 'out << (i+1) << "," << (j+1) << "," << best_ef << "," << best_recall << "\\n";\n            out.flush();'
code = code.replace(old_out, new_out)

with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/research/lookup_5x10.cpp', 'w') as f:
    f.write(code)
