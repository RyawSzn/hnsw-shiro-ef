import re
with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/research/simple_M_moderator.py', 'r') as f:
    code = f.read()

code = code.replace("print(f\"{'-' * 69}\")", "print(f\"{'Avg Recall':<12} | {''}\")\n    print(\"-\" * 84)")
code = code.replace("print(f\"Decile {b:<3} | {len(subset):<12} | {r:>10.4f}\")", "print(f\"Decile {b:<3} | {len(subset):<12} | {subset['recall'].mean():<12.4f} | {r:>10.4f}\")")

with open('/home/ryawszn/dev/cpp/hnsw-2metric-ef/research/simple_M_moderator.py', 'w') as f:
    f.write(code)
