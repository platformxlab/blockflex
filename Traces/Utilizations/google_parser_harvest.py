# In[]

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import os
import mmap
import statistics
import math
from collections import defaultdict





DIR = "task_usage"
"""
task_events.csv format:
task_events/part-?????-of-?????.csv.gz,1,time,INTEGER,YES
task_events/part-?????-of-?????.csv.gz,2,missing info,INTEGER,NO
task_events/part-?????-of-?????.csv.gz,3,job ID,INTEGER,YES
task_events/part-?????-of-?????.csv.gz,4,task index,INTEGER,YES
task_events/part-?????-of-?????.csv.gz,5,machine ID,INTEGER,NO
task_events/part-?????-of-?????.csv.gz,6,event type,INTEGER,YES
task_events/part-?????-of-?????.csv.gz,7,user,STRING_HASH,NO
task_events/part-?????-of-?????.csv.gz,8,scheduling class,INTEGER,NO
task_events/part-?????-of-?????.csv.gz,9,priority,INTEGER,YES
task_events/part-?????-of-?????.csv.gz,10,CPU request,FLOAT,NO
task_events/part-?????-of-?????.csv.gz,11,memory request,FLOAT,NO
task_events/part-?????-of-?????.csv.gz,12,disk space request,FLOAT,NO
task_events/part-?????-of-?????.csv.gz,13,different machines restriction,BOOLEAN,NO

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

#n_ts = 3000000
#utils = [0 for _ in range(n_ts)]
#machines = [set() for _ in range(n_ts)] 
#for f in os.listdir(DIR):
allocs = {}
utils = {}
counts = {}
peak = {}
ids = set()
#print(tasks)
times = {}
n_machines = 0

#Grab the names of the prio tasks
with open("../parsed_all_prio_events.csv", 'r') as f:
    infile = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    for i,line in enumerate(iter(f.readline, b"")):
        if line == '':
            break
        # if len(times) > 4000: break
        temp = line.strip().split(",")
        task_id = int(temp[3])
        uid = temp[2] + ":" + temp[3]
        ts = int(temp[0])//1000000
        if temp[11]:
            disk_space = float(temp[11])
            allocs[uid] = disk_space
            times[uid] = {}

times = {t:{} for _ ,t in enumerate(times) if _ % 20 == 0}
# print(len(times))

#parse the usages
with open("../usages_500.csv", 'r') as f:
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
        if allocs[uid] == 0:
            continue
        ids.add(uid)
        if temp[12]:
            disk_use = float(temp[12])
            times[uid][time_start] = (time_end, disk_use)

        # if time_end > 86400 / 2:
        #     break
        
        if time_end % 300000 == 0 and uid == list(times.keys())[0]:
            print(f"Parsing at timestamp: {time_end}")

# print(len(ids))
    


# with open("harvest_avail_times.txt", "w") as f:
#    for t in ids:
#         if t not in allocs:
#             continue
#         min_time = 1 << 60
#         max_time = 0
#         for k,v in times[t].items():
#             min_time = min(min_time, k);
#             max_time = max(max_time, v[0]);
        
#             print(f"{t},{k},{min_time},{max_time},{allocs[t]}", file = f)
# exit()
## 

# In[]

tot_stats = [[] for _ in range(17)]
to_print = 20
used_machs = set()
max_single = 0
#Currently is not printing to anything, looks like a debug file
ids = list(ids)
with open("harvest_min_max.txt", 'w') as outfile_min_max:
    with open("harvest_avail_times.txt", 'w') as outfile:
        tot_obs = [] 
        tot_ids = set()

        #for t in tasks:
        for t in ids:
            if t not in allocs:
                continue
            prev = -1
            prev_util = -1
            min_util_in_window = float('inf')
            min_util = float('inf')
            max_util = -1
            sum_util = 0
            count = 0

            for i,(k,v) in enumerate(times[t].items()):
                if v[1] > allocs[t]:
                    break
                if prev == -1:
                    prev = k
                    prev_util = 1 - v[1]/allocs[t]
                    min_util_in_window = prev_util
                    continue
                cur = k
                cur_util = 1 - v[1]/allocs[t]

                min_util = min(min_util, cur_util)
                max_util = max(max_util, cur_util)
                sum_util += cur_util * (cur - prev)
                count += cur - prev

                if prev_util > cur_util + 0.15 or prev_util < cur_util - 0.15:
                    if cur - prev >= 3*60*60:
                        print(f"{t},{prev},{cur},{min_util_in_window * allocs[t]},{allocs[t]}", file = outfile)
                        prev = cur
                        prev_util = cur_util
                        min_util_in_window = prev_util
                        continue

                if i == len(times[t]) - 1:
                    print(f"{t},{prev},{cur},{min_util_in_window * allocs[t]},{allocs[t]}", file = outfile)
                
                min_util_in_window = min(min_util_in_window, cur_util)

            if count > 0:
                print(f"{t},{min_util},{sum_util/count},{max_util},{allocs[t]}", file = outfile_min_max)
