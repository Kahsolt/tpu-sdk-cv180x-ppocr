#!/usr/bin/env python3
# Author: Armit
# Create Time: 周四 2024/10/31 

# 将 result.txt 转换为比赛提交格式的 json 文件

import sys
import json
from pathlib import Path

BASE_PATH = Path(__file__).parent
KEY_FILE = BASE_PATH.parent / 'data' / 'ppocr_keys_v1.txt'

fn = sys.argv[1] if len(sys.argv) >= 2 else 'results.txt'

print(f'>> load vocab from {KEY_FILE}')
with open(KEY_FILE, 'r', encoding='utf-8') as fh:
  vocab = fh.read().strip().split('\n')
  vocab.insert(0, None)
print('>> len(vocab):', len(vocab))

print(f'>> load results from {fn}')
with open(fn, 'r', encoding='utf-8') as fh:
  lines = fh.read().strip().split('\n')

results = {}
seg = []
def handle_seg():
  annots = []
  results[seg[0].split('.')[0]] = annots
  for box_ids in seg[1:]:
    box, ids = box_ids.split('|')
    if not ids.strip(): continue
    box = [max(int(e), 0) for e in box.strip().split(' ')]
    ids = [int(e) for e in ids.strip().split(' ')]
    annots.append({
      'points': list(zip(box[::2], box[1::2])),
      'transcription': ''.join([vocab[e] for e in ids]),
    })
  seg.clear()

for line in lines:
  if not line:
    handle_seg()
  else:
    seg.append(line)
if seg: handle_seg()

print('>> save results to results.json')
print('>> len(results):', len(results))
with open('results.json', 'w', encoding='utf-8') as fh:
  json.dump(results, fh, indent=2, ensure_ascii=False)
