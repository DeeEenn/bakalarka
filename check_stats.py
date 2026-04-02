import csv
from collections import Counter

rows = list(csv.DictReader(open('data/video_metadata.csv')))

print(f'\n=== STATISTIKY DATASETU ===')
print(f'Celkem videi: {len(rows)}')

correct = [r for r in rows if r['is_correct'] == '1']
incorrect = [r for r in rows if r['is_correct'] == '0']

print(f'\nSpravne: {len(correct)}')
print(f'Spatne: {len(incorrect)}')

print(f'\nDistribuce error_type:')
et = Counter([r['error_type'] for r in incorrect if r['error_type']])
for k, v in sorted(et.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v}')

print(f'\nDistribuce error_step:')
es = Counter([r['error_step'] for r in incorrect if r['error_step']])
for k, v in sorted(es.items()):
    print(f'  Step {k}: {v}')
