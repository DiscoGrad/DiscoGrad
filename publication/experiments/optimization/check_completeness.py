#!/usr/bin/python3

import os
import sys
import re
from collections import defaultdict

folder = sys.argv[1]

max_rep_id = -1
num_rep_ids = defaultdict(int)

for fname in os.listdir(folder):
  m = re.search('(.+)-rep_(\d\d\d\d)_final_params\.txt', fname)

  if m:
    run_descr, rep_id = m.group(1), int(m.group(2))
    max_rep_id = max(max_rep_id, rep_id)
    num_rep_ids[run_descr] += 1

print(f"expecting {max_rep_id + 1} macroreplications")
for k, v in num_rep_ids.items():
  if v < max_rep_id + 1:
    print(f"{k} is missing {max_rep_id + 1 - v} macroreplications (has {v})")
#$print(os.listdir(folder))
