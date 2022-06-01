import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import os
import mmap
import statistics

DIR = "task_usage"
"""
task_usages.csv format:
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

allocs = {}
utils = {}
counts = {}
peak = {}
mins = {}
ids = set()
times = {}

#Grab the names of the prio tasks from previous parsing
print("Grabbing prio task ids")
with open("parsed_all_prio_events.csv", 'r') as f:
    infile = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    for line in iter(f.readline, b""):
        if line == '':
            break
        temp = line.strip().split(",")
        task_id = int(temp[3])
        uid = temp[2] + ":" + temp[3]
        ts = int(temp[0])//1000000
        if temp[11]:
            disk_space = float(temp[11])
            allocs[uid] = disk_space
            times[uid] = {}

print("Grabbing events for prio tasks")
with open("usages_500.csv", 'r') as f:
    infile = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    for line in iter(f.readline, b""):
        if line == '':
            break
        temp = line.strip().split(",")
        time_start = int(temp[0])//1000000
        time_end = int(temp[1])//1000000
        diff = time_end - time_start
        uid = temp[2] + ":" + temp[3]
        if uid not in times: continue
        ids.add(uid)
        if temp[12]:
            disk_use = float(temp[12])
            if uid not in counts:
                counts[uid] = 0
                utils[uid] = 0
                peak[uid] = 0
                mins[uid] = 1 << 30
            counts[uid] += diff
            utils[uid] += diff * disk_use
            peak[uid] = max(peak[uid], disk_use)
            mins[uid] = min(mins[uid], disk_use)


print("Now printing usage statistics")
with open("google_util_cdf.dat", 'w') as outfile:
    for k,v in utils.items():
        if allocs[k] == 0 or counts[k] == 0:
                continue
        print(f"{k} {utils[k]/counts[k]} {peak[k]} {allocs[k]} {(utils[k]/counts[k])/allocs[k]} {peak[k]/allocs[k]} {mins[k]/allocs[k]}", file=outfile)

