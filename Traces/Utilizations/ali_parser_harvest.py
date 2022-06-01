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


FILE = "../container_usage_sub.csv"


tot_obs = []
tot_ids = set()
ids = set()
max_size = 4000
low_prio_size = 60000
low_count = -1
low_vms = []
n_machines = 6
cur_machine = ""
usages = []
with open(FILE,'r') as infile, open("harvest_min_max.txt", 'w') as outfile_min_max, open("harvest_avail_times.txt", 'w') as outfile, open("low_prio_vm_hybrid.txt", "w") as hybrid_low_prio_file, open("low_prio_vm_times_reparse.txt", "r") as low_prio_file:
            for i, line in enumerate(low_prio_file):
                _id, start, end, util = line.strip().split(",") 
                low_vms += [[_id, int(start), int(end), float(util)]]

            for i, line in enumerate(infile):
                if i % 1000000 == 0: 
                    print(f"Parsed {i} lines")
                #Currently want to average the utilization over the course of the given second
                #Number of unique machines
                container_id, machine_id, time_stamp, cpu_util_percent, mem_util_percent, cpi, mem_gps, mkpi, net_in, net_out, disk_io_percent = line.strip().split(",")
                time_stamp = float(time_stamp)
                if container_id == "":
                    continue

                if cur_machine == "":
                    cur_machine = container_id
                if cur_machine != container_id:
                    if len(usages) > 0 and low_count == -1:
                        prev = usages[0][0]
                        prev_util = 1 - usages[0][1]
                        min_util_in_window = prev_util
                        min_util = prev_util
                        max_util = prev_util
                        sum_util = 0
                        count = 0

                        for i in range(1,len(usages)):
                            cur = usages[i][0]
                            cur_util = 1 - usages[i][1]

                            min_util = min(min_util, cur_util)
                            max_util = max(max_util, cur_util)
                            sum_util += cur_util * (cur - prev)
                            count += cur - prev

                            if prev_util > cur_util + 0.15 or prev_util < cur_util - 0.15:
                                if cur - prev >= 30*60:
                                    print(f"{container_id},{prev},{cur},{min_util_in_window}", file = outfile)
                                    prev = cur
                                    prev_util = cur_util
                                    min_util_in_window = prev_util
                                    continue

                            if i == len(usages) - 1:
                                print(f"{container_id},{prev},{cur},{min_util_in_window}", file = outfile)
                            
                            min_util_in_window = min(min_util_in_window, cur_util)

                        if count > 0:
                            print(f"{container_id},{min_util},{sum_util/count},{max_util}", file = outfile_min_max)

                    if len(usages) > 0 and low_count >= 0: 
                        prev = usages[0][0]
                        prev_util = usages[0][1]
                        sum_util = 0
                        count = 0
                        for i in range(1,len(usages)):
                            cur = usages[i][0]
                            cur_util = usages[i][1]
                            sum_util += cur_util * (cur - prev)
                            count += cur - prev
                            prev = cur
                            prev_util = cur_util
                        if count > 0:
                            for low_index in range(low_count, len(low_vms), low_prio_size):
                                low_vms[low_index][0] = container_id
                                low_vms[low_index][1] %= 738940
                                low_vms[low_index][2] %= 738940
                                if low_vms[low_index][1] >= low_vms[low_index][2]:
                                    low_vms[low_index][1], low_vms[low_index][2] = low_vms[low_index][2], low_vms[low_index][1] 
                                low_vms[low_index][3] = sum_util/count
                                print(",".join(map(str, low_vms[low_index])), file = hybrid_low_prio_file)
                        low_count += 1
                    cur_machine = container_id
                    usages = []

                if disk_io_percent:
                    ids.add(container_id)
                    if (len(ids)) > max_size + low_prio_size:
                        break
                    if len(ids) > max_size and low_count == -1:
                        low_count = 0
                    disk_io_percent = float(disk_io_percent)
                    if disk_io_percent < 0 or disk_io_percent > 100: 
                        continue
                    usages.append((time_stamp, disk_io_percent/100))
                    