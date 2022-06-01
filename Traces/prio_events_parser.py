import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import os
import mmap

DIR = "task_usage"
"""
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
"""

task_cnt = {}
outfile = open("parsed_all_prio_events.csv",'w')
tasks = set()
for i in range(500):
    with open("task_events/part-00%03d-of-00500.csv" % (i), 'r') as f:
        infile = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        print("opened task_events/part-00%03d-of-00500.csv" % (i))
        for line in iter(f.readline, b""):
            if line == '':
                break
            temp = line.strip().split(",")
            task_id = int(temp[3])
            job_id = int(temp[2])
            uid = temp[2] +":"+temp[3]
            if not temp[7]:
                continue
            if int(temp[7]) < 2:
                continue
            prio = int(temp[8])
            if prio < 9:
                continue
            print(line.strip(), file=outfile)
            ts = int(temp[0])//1000000
            if temp[11]:
                tasks.add(uid)
                disk_req = float(temp[11])
                if task_id in task_cnt:
                    task_cnt[task_id] += 1
                else:
                    task_cnt[task_id] = 1
                
outfile.close()
