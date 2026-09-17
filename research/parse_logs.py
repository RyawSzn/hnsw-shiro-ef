import re


def parse_latency(s):
    """Parses a latency string like '324578' or '1.35909e+06' -> float ns"""
    return float(s)


def parse_log(path, system_name):
    """
    Returns:
      baseline_rows: list of dicts {system, dataset, ef, recall, p5, p1, latency_ns}
      method_rows:   list of dicts {system, dataset, avg_latency_ns, avg_recall, p5, p1}
    Only the 'online' section of each dataset (the one containing 'Testing ef:' /
    '=== Final Summary (Median Iteration) ===' blocks) is parsed; the earlier
    'offline'/training section for the same dataset name is ignored automatically
    because it does not contain those markers.
    """
    with open(path, "r", errors="ignore") as f:
        text = f.read()

    lines = text.split("\n")

    baseline_rows = []
    method_rows = []
    current_dataset = None

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        m = re.match(r"Dataset:\s*(\S+)", line)
        if m:
            current_dataset = m.group(1)
            i += 1
            continue

        # Adaptive ("method") result block
        if line.strip() == "=== Final Summary (Median Iteration) ===":
            block = "\n".join(lines[i : i + 8])
            lat_m = re.search(r"Average Latency:\s*([0-9.eE+\-]+)\s*ns", block)
            rec_m = re.search(r"Average Recall:\s*([0-9.]+)", block)
            p5_m = re.search(r"5th percentile recall:\s*([0-9.]+)", block)
            p1_m = re.search(r"1st percentile recall:\s*([0-9.]+)", block)
            if lat_m and rec_m:
                method_rows.append(
                    {
                        "system": system_name,
                        "dataset": current_dataset,
                        "avg_latency_ns": parse_latency(lat_m.group(1)),
                        "avg_recall": float(rec_m.group(1)),
                        "p5_recall": float(p5_m.group(1)) if p5_m else None,
                        "p1_recall": float(p1_m.group(1)) if p1_m else None,
                    }
                )
            i += 1
            continue

        # Baseline fixed-ef sweep block
        m = re.match(r"Testing ef:\s*(\d+)", line)
        if m:
            ef = int(m.group(1))
            block = "\n".join(lines[i : i + 12])
            rec_m = re.search(r"Average Recall:\s*([0-9.]+)", block)
            p5_m = re.search(r"5th percentile recall:\s*([0-9.]+)", block)
            p1_m = re.search(r"1st percentile recall:\s*([0-9.]+)", block)
            lat_m = re.search(r"Average Latency:\s*([0-9.eE+\-]+)\s*ns", block)
            if rec_m and lat_m:
                baseline_rows.append(
                    {
                        "system": system_name,
                        "dataset": current_dataset,
                        "ef": ef,
                        "recall": float(rec_m.group(1)),
                        "p5_recall": float(p5_m.group(1)) if p5_m else None,
                        "p1_recall": float(p1_m.group(1)) if p1_m else None,
                        "latency_ns": parse_latency(lat_m.group(1)),
                    }
                )
            i += 1
            continue

        i += 1

    return baseline_rows, method_rows


if __name__ == "__main__":
    b1, m1 = parse_log("research/log/output_ada_full.log", "ada")
    b2, m2 = parse_log("research/log/output_shiro_full.log", "shiro")
    print(f"ada baseline points: {len(b1)}, method rows: {len(m1)}")
    print(f"shiro baseline points: {len(b2)}, method rows: {len(m2)}")
    print("ada datasets:", sorted(set(r["dataset"] for r in b1)))
    print("shiro datasets:", sorted(set(r["dataset"] for r in b2)))
