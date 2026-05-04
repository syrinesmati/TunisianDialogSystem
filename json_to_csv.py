import json
import csv

# Read the JSON file
with open('llm/sft/data/raw/categorized.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Get all unique keys from the data
all_keys = set()
for item in data:
    all_keys.update(item.keys())

# Sort keys for consistent ordering
fieldnames = sorted(list(all_keys))

# Write to CSV file
output_file = 'llm/sft/data/raw/categorized.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print(f'Successfully converted JSON to CSV')
print(f'Output file: {output_file}')
print(f'Total rows: {len(data)}')
print(f'Columns: {", ".join(fieldnames)}')
