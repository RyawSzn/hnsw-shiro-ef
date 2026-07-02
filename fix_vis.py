with open("research/visualize_ef_map.py", "r") as f:
    text = f.read()

# remove build_links
if "def build_links(" in text:
    pass # Wait, let's just write the full python file because it's small.
