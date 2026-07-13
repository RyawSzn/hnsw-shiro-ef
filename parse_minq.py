import re

curr_ds = None
curr_minq = None

re_dataset = re.compile(r"^Dataset:\s+(\S+)")
re_minq = re.compile(r"^---\s+min_queries_per_score:\s+(\d+)")
re_results = re.compile(r"^\s*\d+\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,")

print(f"{'Dataset':<25} {'min_q':>8} {'Avg Rec':>10} {'p5 Rec':>10} {'p1 Rec':>10} {'Time(ms)':>9}")
print("-" * 78)

with open('ablation_cv.log', 'r') as f:
    for line in f:
        line = line.strip()
        
        m = re_dataset.search(line)
        if m:
            curr_ds = m.group(1)
            
        m = re_minq.search(line)
        if m:
            curr_minq = int(m.group(1))
            
        m = re_results.search(line)
        if m and curr_ds and curr_minq is not None:
            time = int(m.group(1))
            avg = float(m.group(2))
            p5 = float(m.group(3))
            p1 = float(m.group(4))
            print(f"{curr_ds:<25} {curr_minq:>8} {avg:>10.4f} {p5:>10.4f} {p1:>10.4f} {time:>9}")
