import sys

with open('/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/visualize_ef_map.py', 'r') as f:
    content = f.read()

# Make the lines plotting conditional
old_plot_lines = """    for c in dep_centers:
        plt.axhline(
            y=c,
            color="r",
            linestyle="--",
            alpha=0.5,
            label="Center" if c == dep_centers[0] else "",
        )"""

new_plot_lines = """    # Only plot center lines if there aren't too many (e.g. don't clutter if n_cv_tables=0 creates 101 centers)
    if len(dep_centers) <= 20:
        for c in dep_centers:
            plt.axhline(
                y=c,
                color="r",
                linestyle="--",
                alpha=0.5,
                label="Center" if c == dep_centers[0] else "",
            )
    else:
        print(f"Skipping drawing {len(dep_centers)} center lines to prevent cluttering the heatmap.")"""

content = content.replace(old_plot_lines, new_plot_lines)

with open('/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/visualize_ef_map.py', 'w') as f:
    f.write(content)
