import sys
import numpy as np
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
ids = set()
#print(tasks)
times = {}

time_min = {}
time_max = {}
n_machines = 0
disks = list()

#Grab the names of the prio tasks
with open("parsed_all_prio_events.csv", 'r') as f:
    infile = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    for line in iter(f.readline, b""):
        if line == '':
            break
        temp = line.strip().split(",")
        task_id = int(temp[3])
        uid = temp[2] + ":" + temp[3]
        ts = int(temp[0])//1000000
        ids.add(uid)
        if temp[11]:
            disk_space = float(temp[11])
            allocs[uid] = disk_space
            disks.append(disk_space)
            times[uid] = {}


to_print = 10
outs = 16
divs = 1/outs * 100
#Currently is not printing to anything, looks like a debug file
with open("sample_tasks_500.txt", 'w') as outfile:
    #for t in tasks:
    for t in ids:
        if t not in peak or t not in allocs:
            continue
        #print(f"{t} {peak[t]} {allocs[t]}", file=outfile)
        vals = []
        stats = [[] for _ in range(17)]
        all_obs = []
        tot = 0
        stats_all = 0
        prev = -1
        prev_util = -1
        for k,v in times[t].items():
            #print(f"{k} {v[0]} {v[1]}", file=outfile)
            if prev == -1:
                prev = k
                prev_util = int(((v[1]*100)/allocs[t])//divs)
                continue
            cur = k
            cur_util = int(((v[1]*100)/allocs[t])//divs)
            if prev_util != cur_util:
                if int(prev_util) > 16: 
                    prev_util = 16
                stats[int(prev_util)].append(cur - prev)
                all_obs.append([cur-prev, prev_util])
                stats_all += cur - prev
                tot += 1
                prev = cur
                prev_util = cur_util
        if tot > 10: 
            print("Printing: " + str(t));
            with open("outs/" + t + ".txt", "w") as m_file:
                print(f"total_obs: {tot}, avg_duration: {stats_all/tot}",file=m_file)
                for obs in all_obs:
                    print(f"{obs[0]} {obs[1]}", file=m_file)
            to_print-=1
            if to_print == 0:
                exit(0)
