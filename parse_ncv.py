import re

curr_ds = None
curr_ncv = None

re_dataset = re.compile(r"^Dataset:\s+(\S+)")
re_ncv = re.compile(r"^---\s+n_cv_tables:\s+(\d+)")
re_results = re.compile(r"^\s*\d+\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,")

print(f"{'Dataset':<25} {'n_cv':>8} {'Avg Rec':>10} {'p5 Rec':>10} {'p1 Rec':>10} {'Time(ms)':>9}")
print("-" * 78)

with open('ablation_cv.log', 'r') as f:
    for line in f:
        line = line.strip()
        
        m = re_dataset.search(line)
        if m:
            curr_ds = m.group(1)
            
        m = re_ncv.search(line)
        if m:
            curr_ncv = int(m.group(1))
            
        m = re_results.search(line)
        if m and curr_ds and curr_ncv is not None:
            time = int(m.group(1))
            avg = float(m.group(2))
            p5 = float(m.group(3))
            p1 = float(m.group(4))
            print(f"{curr_ds:<25} {curr_ncv:>8} {avg:>10.4f} {p5:>10.4f} {p1:>10.4f} {time:>9}")
