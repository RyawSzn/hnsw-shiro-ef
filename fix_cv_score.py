import re

with open('hnswlib/adaptive_ef.h', 'r') as f:
    code = f.read()

code = code.replace(
    'int cv_score = std::max(0, std::min(100, static_cast<int>(cvs[i] * 100.0f)));',
    'int cv_score = std::max(0, std::min(100, static_cast<int>(cvs[i])));'
)

with open('hnswlib/adaptive_ef.h', 'w') as f:
    f.write(code)

