with open('experiments_driver/run.cpp', 'r') as f:
    content = f.read()

content = content.replace("    offline_exp();", "    // offline_exp();")
content = content.replace("    online_exp();", "    // online_exp();")

with open('experiments_driver/run.cpp', 'w') as f:
    f.write(content)
