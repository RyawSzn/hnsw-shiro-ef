import struct
import sys
import matplotlib.pyplot as plt
import numpy as np

def read_pod(f, fmt):
    sz = struct.calcsize(fmt)
    data = f.read(sz)
    if len(data) < sz:
        raise EOFError
    return struct.unpack(fmt, data)[0]

def read_table(f):
    sz = read_pod(f, 'Q')
    table = []
    for _ in range(sz):
        score = read_pod(f, 'i')
        recall_size = read_pod(f, 'Q')
        recalls = []
        for _ in range(recall_size):
            ef = read_pod(f, 'i')
            recall = read_pod(f, 'f')
            recalls.append((ef, recall))
        table.append((score, recalls))
    return table

def deserialize(filename):
    with open(filename, 'rb') as f:
        ef_recall_estimators = read_table(f)
        expected_recall = read_pod(f, 'f')
        wae = read_pod(f, 'f')
        
        n_dep = read_pod(f, 'Q')
        dep_tables = []
        for _ in range(n_dep):
            dep_tables.append(read_table(f))
            
        n_thresh = read_pod(f, 'Q')
        dep_centers = []
        for _ in range(n_thresh):
            dep_centers.append(read_pod(f, 'f'))
            
    return ef_recall_estimators, expected_recall, wae, dep_tables, dep_centers

def build_links(table):
    links = [0] * 101
    idx = 0
    for i in range(101):
        while idx < len(table) and table[idx][0] < i:
            idx += 1
            
        a_index = idx - 1 if idx > 0 else -1
        b_index = idx if idx < len(table) else -1
        
        if a_index != -1 and b_index != -1:
            if abs(table[a_index][0] - i) <= abs(table[b_index][0] - i):
                links[i] = a_index
            else:
                links[i] = b_index
        elif a_index != -1:
            links[i] = a_index
        else:
            links[i] = b_index
    return links

def lookup_ef(table, links, score, expected_recall):
    clamped = max(0, min(100, int(score)))
    idx = links[clamped]
    ef_recalls = table[idx][1]
    for ef, recall in ef_recalls:
        if recall >= expected_recall:
            return ef
    return ef_recalls[-1][0]

def smoothed_ef(table, links, score, expected_recall):
    first = lookup_ef(table, links, score, expected_recall)
    if score < 1 or score >= 100:
        return first
    return (first + lookup_ef(table, links, score - 1, expected_recall) + lookup_ef(table, links, score + 1, expected_recall)) / 3.0

def estimate_ef2(tables, all_links, dep_centers, score, d_ep, expected_recall):
    n_centers = len(dep_centers)
    if n_centers == 1:
        return smoothed_ef(tables[0], all_links[0], score, expected_recall)
        
    if d_ep <= dep_centers[0]:
        return smoothed_ef(tables[0], all_links[0], score, expected_recall)
    if d_ep >= dep_centers[-1]:
        return smoothed_ef(tables[-1], all_links[-1], score, expected_recall)
        
    idx = 0
    while idx < n_centers - 1 and d_ep > dep_centers[idx + 1]:
        idx += 1
        
    c0 = dep_centers[idx]
    c1 = dep_centers[idx + 1]
    w = (d_ep - c0) / (c1 - c0) if c1 != c0 else 0
    
    ef0 = smoothed_ef(tables[idx], all_links[idx], score, expected_recall)
    ef1 = smoothed_ef(tables[idx + 1], all_links[idx + 1], score, expected_recall)
    
    return int(ef0 * (1.0 - w) + ef1 * w + 0.5)

def plot_heatmap(filename, output_png):
    _, expected_recall, _, dep_tables, dep_centers = deserialize(filename)
    
    if not dep_tables:
        print("No dependency tables found.")
        return
        
    all_links = [build_links(t) for t in dep_tables]
    
    scores = np.linspace(0, 100, 101)
    
    d_ep_min = dep_centers[0] - (dep_centers[-1] - dep_centers[0]) * 0.1
    d_ep_max = dep_centers[-1] + (dep_centers[-1] - dep_centers[0]) * 0.1
    
    if d_ep_min == d_ep_max:
        d_ep_min -= 1.0
        d_ep_max += 1.0
        
    d_eps = np.linspace(d_ep_min, d_ep_max, 100)
    
    X, Y = np.meshgrid(scores, d_eps)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = estimate_ef2(dep_tables, all_links, dep_centers, X[i, j], Y[i, j], expected_recall)
            
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Estimated ef')
    
    for c in dep_centers:
        plt.axhline(y=c, color='r', linestyle='--', alpha=0.5, label='Center' if c == dep_centers[0] else "")
        
    plt.xlabel('Revisit Rank Score')
    plt.ylabel('d_ep - lowerBound')
    plt.title(f'Estimated ef mapped on Score / d_ep Coordinate Plane\nExpected Recall: {expected_recall}')
    plt.legend()
    plt.savefig(output_png)
    print(f"Heatmap saved to {output_png}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ef_adaptor.bin> <output.png>")
        sys.exit(1)
    plot_heatmap(sys.argv[1], sys.argv[2])
