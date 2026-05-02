#!/usr/bin/env python3
import json
import os
import re
from collections import Counter


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def classify_text(text, rules):
    t = text.lower()
    for pattern, cat in rules:
        if re.search(pattern, t):
            return cat
    return None


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    json_path = os.path.normpath(os.path.join(script_dir, '..', 'data', 'raw', 'random.json'))

    data = load_json(json_path)

    # Ordered rules: regex pattern -> Arabic category
    rules = [
        (r'مرصد حضري|المرصد الحضري|الرياض', 'تخطيط حضري'),
        (r'حوت|محيط|بحر|سمك|حيتان|زعانف', 'أحياء بحرية'),
        (r'عملية|جيش|نابلس|مسلح|قصف|هجوم|عسكر|اعتقال|تقبض', 'تاريخ عسكري'),
        (r'أذان|الإقامة|صلاة|مؤذن|مسجد|الإسلام|دين|الشرعي|قد قامت الصلاة|حي على الصلاة', 'دين'),
        (r'لاعب|فريق|بطولة|كاس|دورى|ماتش|هدف|ترجي|نادي|كرة', 'رياضة'),
        (r'تكامل|فيزيا|ميكلسون|أثير|نظرية|تجرب', 'علوم'),
        (r'جامع|زيتونة|أُثري|تاريخ|تأسس|مدينة|آثار|يونسكو|موقع', 'ثقافة'),
        (r'سكان|معدل|عدد السكان|ولاية|مقاطعة|بنسيلفانيا|بورتاج', 'عام'),
    ]

    counts = Counter()
    updated = 0
    for obj in data:
        # If already has an Arabic category and it's non-empty, keep it
        if 'category' in obj and isinstance(obj['category'], str) and obj['category'].strip():
            # If category is in English, attempt to classify and replace with Arabic
            # We'll check if it contains non-ASCII letters to guess Arabic
            if all(ord(c) < 128 for c in obj['category']):
                # try to classify from instruction/response
                cat = None
                instr = obj.get('instruction', '')
                resp = obj.get('response', '')
                cat = classify_text(instr + ' ' + resp, rules)
                if cat is None:
                    cat = 'عام'
                obj['category'] = cat
                counts[cat] += 1
                updated += 1
            else:
                counts[obj['category']] += 1
        else:
            instr = obj.get('instruction', '')
            resp = obj.get('response', '')
            cat = classify_text(instr + ' ' + resp, rules)
            if cat is None:
                cat = 'عام'
            obj['category'] = cat
            counts[cat] += 1
            updated += 1

    save_json(json_path, data)

    print(f'Updated {updated} entries in {json_path}')
    print('Category counts:')
    for k, v in counts.most_common():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
