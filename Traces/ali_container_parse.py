import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import statistics
import math



FILE = "container_usage_sub.csv"
"""
machine_usage.csv format:
+------------------------------------------------------------------------------------------+
| Field            | Type       | Label | Comment                                          |
+------------------------------------------------------------------------------------------+
| machine_id       | string     |       | uid of machine                                   |
| time_stamp       | double     |       | time stamp, in second                            |
| cpu_util_percent | bigint     |       | [0, 100]                                         |
| mem_util_percent | bigint     |       | [0, 100]                                         |
| mem_gps          | double     |       |  normalized memory bandwidth, [0, 100]           |
| mkpi             | bigint     |       |  cache miss per thousand instruction             |
| net_in           | double     |       |  normalized in coming network traffic, [0, 100] |
| net_out          | double     |       |  normalized out going network traffic, [0, 100] |
| disk_io_percent  | double     |       |  [0, 100], abnormal values are of -1 or 101      |
+------------------------------------------------------------------------------------------+

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
| net_in           | double     |       | normalized in coming network traffic, [0, 100] |
| net_out          | double     |       | normalized out going network traffic, [0, 100] |
| disk_io_percent  | double     |       | [0, 100], abnormal values are of -1 or 101      |
+-----------------------------------------------------------------------------------------+
"""


f = open("ali_stats_util_cdf.txt", "w")

percentages = [90,75,50]
HOUR = 3600
DAY = 24*3600
harvest_length = [HOUR, 6*HOUR, 0.5*DAY, 1*DAY, 3*DAY]
harvest_interval = 10
for perc in percentages:
    for length in harvest_length:
        tot_obs = []
        all_obs = []
        tot_ids = set()
        total = {}
        count = {}
        max_util = {}
        min_util = {}
        ids = set()
        max_size = 4000
        n_machines = 6
        cur_machine = ""
        usages = []
        harvest_rate = {_:0 for _ in range(0, 9*DAY, harvest_interval)}
        with open(FILE,'r') as infile:
            infile.seek(0)
            for i, line in enumerate(infile):
                container_id, machine_id, time_stamp, cpu_util_percent, mem_util_percent, cpi, mem_gps, mkpi, net_in, net_out, disk_io_percent = line.strip().split(",")
                time_stamp = float(time_stamp)
                if container_id == "":
                    continue
                if cur_machine == "":
                    cur_machine = container_id
                if cur_machine != container_id:
                    if len(usages) > 0:
                        prev = usages[0][0]
                        prev_util = int(usages[0][1])
                        for i in range(1,len(usages)):
                            cur = usages[i][0]
                            cur_util = int(usages[i][1])
                            #print(cur_util,file=f)
                            if i != len(usages)-1 and cur_util == prev_util:
                                continue
                            #Both above and below are removable 
                            if cur_util >= perc and prev_util >= perc: continue
                            if cur_util < perc and prev_util < perc:
                                if i == len(usages) - 1:
                                    tot_obs.append(cur - prev)
                                    all_obs.append((prev, cur))
                                    tot_ids.add(container_id)
                                continue
                            #print(f"{cur - prev} {prev_util}",file=f)
                            if prev < cur:
                                if cur - prev <= 180:
                                    continue
                                if cur_util >= perc:
                                    tot_obs.append(cur - prev)
                                    all_obs.append((prev, cur))
                                    tot_ids.add(container_id)
                                    prev = cur
                                    prev_util = cur_util
                                    #ind = int(prev_util)
                                    #stats[ind].append(cur - prev)
                                else:
                                    prev = cur
                                    prev_util = cur_util


                            #prev = cur
                            #prev_util = cur_util
                    cur_machine = container_id
                    usages = []

                if disk_io_percent:
                    ids.add(container_id)
                    if (len(ids)) > max_size:
                        break
                    disk_io_percent = float(disk_io_percent)
                    if disk_io_percent < 0 or disk_io_percent > 100: 
                        continue
                    usages.append((time_stamp, disk_io_percent))
                    if time_stamp not in total:
                        total[time_stamp] = disk_io_percent
                        count[time_stamp] = 1
                        max_util[time_stamp] = disk_io_percent
                        min_util[time_stamp] = disk_io_percent
                    else:
                        total[time_stamp] += disk_io_percent
                        count[time_stamp] += 1
                        max_util[time_stamp] = max(max_util[time_stamp], disk_io_percent)
                        min_util[time_stamp] = min(min_util[time_stamp], disk_io_percent)

        print(f"--------------Percentage:{100-perc}% Harvest_length:{length}------------------------",file=f)
        print(f"--------------Percentage:{100-perc}% Harvest_length:{length}------------------------") 

        for prev, cur in all_obs:
            for timestep in range(int(prev), int(cur - length), harvest_interval):
                harvest_rate[timestep] += 1
        
        print(f"{harvest_rate}",file=f)

        f.flush()

            


f.close()

