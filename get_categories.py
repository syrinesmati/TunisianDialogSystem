import json
from collections import Counter

# Load the filtered data
with open('llm/sft/data/raw/categorized.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract all unique categories
categories = set()
category_count = Counter()

for item in data:
    if 'category' in item:
        category = item['category']
        categories.add(category)
        category_count[category] += 1

# Sort categories alphabetically
sorted_categories = sorted(categories)

# Prepare output
output_lines = []
output_lines.append("=" * 50)
output_lines.append("ALL CATEGORIES IN FILTERED_DATA.JSON")
output_lines.append("=" * 50)
output_lines.append(f"\nTotal unique categories: {len(sorted_categories)}\n")

for i, category in enumerate(sorted_categories, 1):
    output_lines.append(f"{i}. {category} ({category_count[category]} entries)")

output_lines.append("\n" + "=" * 50)
output_lines.append("CATEGORY LIST:")
output_lines.append("=" * 50)
for category in sorted_categories:
    output_lines.append(f"- {category}")

# Print to console
for line in output_lines:
    print(line)

# Write to file
with open('categories_list.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"\n✓ Categories saved to 'categories_list.txt'")
