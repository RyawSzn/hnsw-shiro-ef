import re

with open('./hnswlib/shiro_ef.h', 'r') as f:
    content = f.read()

pattern = r'// Linear Interpolation for Holes.*?efs\[s\] = left_ef \* \(1\.0f - w\) \+ right_ef \* w;'

replacement = """// LDW (Inverse Distance Weighting) for Holes
                            float sum_w = 0.0f;
                            float sum_ef = 0.0f;
                            for (auto const& pair : score_to_ef) {
                                float dist = std::abs(s - pair.first);
                                float w = 1.0f / (dist * dist); // Inverse distance squared
                                sum_w += w;
                                sum_ef += w * pair.second;
                            }
                            efs[s] = (sum_w > 0.0f) ? (sum_ef / sum_w) : 0.0f;"""

new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

if new_content != content:
    with open('./hnswlib/shiro_ef.h', 'w') as f:
        f.write(new_content)
    print("Success")
else:
    print("Regex failed")
