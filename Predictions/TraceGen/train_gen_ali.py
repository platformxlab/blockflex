import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import statistics

FILE = "container_usage.csv"
"""
container_usage.csv format:
+-----------------------------------------------------------------------------------------+
| container_id     | string     |       | uid of a container                              |
| machine_id       | string     |       | uid of container's host machine                 |
| time_stamp       | double     |       | time stamp, in second                           |
| cpu_util_percent | bigint     |       |                                                 |
| mem_util_percent | bigint     |       |                                                 |
| cpi              | double     |       |                                                 |
| mem_gps          | double     |       | normalized memory bandwidth, [0, 100]           |
| mpki             | bigint     |       |                                                 |
| net_in           | double     |       | normarlized in coming network traffic, [0, 100] |
| net_out          | double     |       | normarlized out going network traffic, [0, 100] |
| disk_io_percent  | double     |       | [0, 100], abnormal values are of -1 or 101      |
+-----------------------------------------------------------------------------------------+
"""
infile = open(FILE, 'r')

ids = set()
max_size = 30000
cur_machine = ""
usages = []
to_print = 10

for line in infile:
    #Currently want to average the utilization over the course of the given second
    #Number of unique machines
    container_id, machine_id, time_stamp, cpu_util_percent, mem_util_percent, cpi, mem_gps, mkpi, net_in, net_out, disk_io_percent = line.strip().split(",")
    time_stamp = float(time_stamp)
    if container_id == "":
        cur_machine = container_id
    if cur_machine != container_id:
        if len(usages) > 0:
            stats = [ [] for _ in range(17)]
            stats_all = 0
            all_obs = []
            alpha = 0.2
            prev = usages[0][0]
            prev_util = usages[0][1]
            tot = 0
            diff = 0
            for i in range(1,len(usages)):
                cur = usages[i][0]
                cur_util = usages[i][1]
                if i != len(usages)-1 and cur_util == prev_util:
                    continue
                #print(f"{cur - prev} {prev_util}",file=f)
                if prev < cur:
                    if cur - prev <= 300:
                        continue
                    ind = int(prev_util)
                    stats[ind].append(cur - prev)
                    stats_all += cur - prev
                    all_obs.append([cur - prev, prev_util])
                    tot += 1
                prev = cur
                prev_util = cur_util
            if tot > 250:
                with open(cur_machine + ".txt", "w") as f:
                    print(f"total_obs: {tot} diff:{diff} avg_duration: {stats_all/tot}",file=f)
                    for obs in all_obs:
                        print(f"{obs[0]} {obs[1]}", file=f)
                to_print-=1
                if to_print==0: 
                    exit(0)
        cur_machine = container_id
        usages = []
    if disk_io_percent:
        ids.add(container_id)
        if (len(ids)) > max_size:
            break
        disk_io_percent = float(disk_io_percent)
        if disk_io_percent < 0 or disk_io_percent > 100: 
            continue
        usages.append((time_stamp, disk_io_percent//6.25))
infile.close()
