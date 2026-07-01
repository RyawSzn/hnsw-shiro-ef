with open('hnswlib/adaptive_ef.h', 'r') as f:
    content = f.read()

if '#include <map>' not in content:
    content = '#include <map>\n' + content
    
    with open('hnswlib/adaptive_ef.h', 'w') as f:
        f.write(content)
