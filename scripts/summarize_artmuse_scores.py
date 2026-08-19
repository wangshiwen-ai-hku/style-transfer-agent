#!/usr/bin/env python3
import os
import json
import csv
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'result_compare', 'artmuse'))
OUT_CSV = os.path.join(ROOT, 'summary_scores_artmuse.csv')

metric_keys = ['Content Preservation', 'Style Faithfulness', 'Artifact Quality', 'Overall Score']

def summarize_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sums = defaultdict(float)
    counts = defaultdict(int)
    aest_sum = 0.0
    aest_count = 0
    for item_key, item_val in data.items():
        scores = item_val.get('Scores') or {}
        for k in metric_keys:
            if k in scores and isinstance(scores[k], (int, float)):
                sums[k] += float(scores[k])
                counts[k] += 1
        # AestheticScore (optional)
        if 'AestheticScore' in item_val and isinstance(item_val['AestheticScore'], (int, float)):
            aest_sum += float(item_val['AestheticScore'])
            aest_count += 1
    return sums, counts, aest_sum, aest_count


def main():
    results = {}
    # find json files under ROOT
    for dirpath, dirnames, filenames in os.walk(ROOT):
        for fn in filenames:
            if not fn.lower().endswith('.json'):
                continue
            fullpath = os.path.join(dirpath, fn)
            model_name = os.path.basename(dirpath)
            sums, counts, aest_sum, aest_count = summarize_json_file(fullpath)
            # accumulate per model (in case multiple files per model directory)
            if model_name not in results:
                results[model_name] = {'sums': defaultdict(float), 'counts': defaultdict(int), 'aest_sum':0.0, 'aest_count':0}
            r = results[model_name]
            for k,v in sums.items():
                r['sums'][k] += v
            for k,v in counts.items():
                r['counts'][k] += v
            r['aest_sum'] += aest_sum
            r['aest_count'] += aest_count

    # compute averages and write CSV
    rows = []
    for model, r in sorted(results.items()):
        row = {
            'Model': model,
        }
        for k in metric_keys:
            if r['counts'].get(k,0) > 0:
                avg = r['sums'][k] / r['counts'][k]
                row[k] = round(avg, 2)
            else:
                row[k] = ''
        if r['aest_count'] > 0:
            row['AestheticScore'] = round(r['aest_sum'] / r['aest_count'], 2)
        else:
            row['AestheticScore'] = ''
        rows.append(row)

    # write CSV
    fieldnames = ['Model'] + metric_keys + ['AestheticScore']
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # print summary
    print(f'Wrote summary CSV to: {OUT_CSV}')
    for r in rows:
        print(r)

if __name__ == '__main__':
    main()
