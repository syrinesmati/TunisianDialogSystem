import re

with open('llm/sft/data/raw/filtered_data.json', encoding='utf-8') as f:
    data = f.read()

# Count objects with "id" field
count = len(re.findall(r'"id"', data))
print(f'Total entries: {count}')
