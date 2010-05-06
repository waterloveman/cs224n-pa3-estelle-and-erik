#!/usr/bin/python

import sys

for line in sys.stdin:
    data = line[:-1].split('\t')
    if len(data) != 3: continue
    if data[1] != data[2]: print ','.join(data)
