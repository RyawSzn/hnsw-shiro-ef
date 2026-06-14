with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/CMakeLists.txt', 'r') as f:
    lines = f.readlines()

# Clean up broken block at the end
clean_lines = []
for line in lines:
    if line.strip() == "${HDF5_LIBRARIES}" and clean_lines[-1].strip() == ")":
        continue
    if line.strip() == ")" and clean_lines[-1].strip() == ")":
        continue
    clean_lines.append(line)

with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/CMakeLists.txt', 'w') as f:
    f.writelines(clean_lines)
