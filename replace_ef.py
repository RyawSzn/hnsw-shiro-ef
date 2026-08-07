import re

with open('./hnswlib/shiro_ef.h', 'r') as f:
    content = f.read()

# Add the constexpr at the top (after includes)
if 'constexpr int WAE_CALC_METHOD' not in content:
    content = re.sub(
        r'(#pragma once\n)',
        r'\1\nconstexpr int WAE_CALC_METHOD = 0;\n',
        content,
        count=1
    )

old_pattern = r"(\s+size_t ef = out_table\[i\]\.second\.back\(\)\.first;)\s+for\s*\(size_t j = 0; j < out_table\[i\]\.second\.size\(\) - 1;\s*\+\+j\)\s*\{\s*if\s*\(out_table\[i\]\.second\[j\]\.second >= expected_recall\)\s*\{\s*ef = out_table\[i\]\.second\[j\]\.first;\s*break;\s*\}\s*\}"

def replacer(match):
    prefix = match.group(1)
    # The indentation matches the prefix's indentation
    indent = prefix.split('size_t')[0]
    return f"""{prefix}
{indent}if constexpr (WAE_CALC_METHOD == 0) {{
{indent}    for (size_t j = 0; j < out_table[i].second.size() - 1; ++j) {{
{indent}        if (out_table[i].second[j].second >= expected_recall) {{
{indent}            ef = out_table[i].second[j].first;
{indent}            break;
{indent}        }}
{indent}    }}
{indent}}} else if constexpr (WAE_CALC_METHOD == 1) {{
{indent}    float sum_ef = 0;
{indent}    int valid_count = 0;
{indent}    for (size_t j = 0; j < out_table[i].second.size(); ++j) {{
{indent}        if (out_table[i].second[j].second >= expected_recall) {{
{indent}            sum_ef += out_table[i].second[j].first;
{indent}            valid_count++;
{indent}        }}
{indent}    }}
{indent}    if (valid_count > 0) {{
{indent}        ef = std::round(sum_ef / valid_count);
{indent}    }}
{indent}}}"""

new_content = re.sub(old_pattern, replacer, content)

with open('./hnswlib/shiro_ef.h', 'w') as f:
    f.write(new_content)

print(f"Replaced {content.count('size_t ef = out_table[i].second.back().first;')} instances.")
print(f"File changed: {content != new_content}")
