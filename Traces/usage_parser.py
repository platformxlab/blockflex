import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import os
import mmap

DIR = "task_usage"
"""
task_events.csv format:
field number,content,format,mandatory
1,start time,INTEGER,YES
2,end time,INTEGER,YES
3,job ID,INTEGER,YES
4,task index,INTEGER,YES
5,machine ID,INTEGER,YES
6,CPU rate,FLOAT,NO
7,canonical memory usage,FLOAT,NO
8,assigned memory usage,FLOAT,NO
9,unmapped page cache,FLOAT,NO
10,total page cache,FLOAT,NO
11,maximum memory usage,FLOAT,NO
12,disk I/O time,FLOAT,NO
13,local disk space usage,FLOAT,NO
14,maximum CPU rate,FLOAT,NO
15,maximum disk IO time,FLOAT,NO
16,cycles per instruction,FLOAT,NO
17,memory accesses per instruction,FLOAT,NO
18,sample portion,FLOAT,NO
19,aggregation type,BOOLEAN,NO
20,sampled CPU usage,FLOAT,NO
"""

outfile = open("usages_500.csv",'w')
task_cnt = {}
ids = set()
with open("parsed_all_prio_events.csv", 'r') as f:
    infile = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    for line in iter(f.readline, b""):
        if line == '':
            break
        temp = line.strip.split(",")
        uid = temp[2]+":"+temp[3]
        ids.add(uid)

for i in range(500):
    with open("task_usage/part-00%03d-of-00500.csv" % (i), 'r') as f:
        infile = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        print("opened task_usage/part-00%03d-of-00500.csv" % (i))
        for line in iter(f.readline, b""):
            if line == '':
                break
            temp = line.strip().split(",")
            uid = temp[2] + ":" + temp[3]
            if uid not in ids:
                continue
            print(line.strip(), file=outfile)

outfile.close()
