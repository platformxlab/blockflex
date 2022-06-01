#!/bin/python3

import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from macros import *

plt.rcParams.update({'font.family': 'serif'})

harvest = []
no_harvest = []
unsold = []

INTERVAL = 1
PAGE_SZ = 16*KB 

scheduling_interval = 10
workload = sys.argv[1]
TRACE_BW_FILE = f"{USR_DIR}/traces/{workload}_bw.npy"
DIR = f"{USR_DIR}/results/{workload}"
bw_traces = np.load(TRACE_BW_FILE)
kernel_size = 60
kernel = np.ones(kernel_size) / kernel_size

filenames = ['no_harvest_bw', 'unsold_harvest_bw', 'harvest_bw']
for filename in filenames:
    with open(f'{DIR}/{filename}.out', 'r') as f:
        mode = None
        regular = []
        for i, row in enumerate(f):
            row = row.strip()
            if row == "PRINTING REGULAR":
                mode = 'r'
            elif row == "PRINTING HARVEST":
                mode = 'h'
            else:
                if mode == 'r':
                    reads, writes = list(map(int, row.split(" ")))
                    regular.append((reads + writes)*PAGE_SZ/INTERVAL)
                elif mode == 'h':
                    reads, writes = list(map(int, row.split(" ")))
                    if filename == "harvest_bw":
                        harvest.append((reads + writes)*PAGE_SZ/INTERVAL)
                    elif filename == "no_harvest_bw":
                        no_harvest.append((reads + writes)*PAGE_SZ/INTERVAL)
                    elif filename == "unsold_harvest_bw":
                        unsold.append((reads + writes)*PAGE_SZ/INTERVAL)
        
        regular = np.array(regular) / MB

harvest_start = len(no_harvest) // 2
harvest = np.array(harvest) / MB
no_harvest = np.array(no_harvest) / MB
harvest = np.convolve(harvest, kernel, mode='same')
no_harvest = np.convolve(no_harvest, kernel, mode='same')
unsold = np.array(unsold) / MB
unsold = np.convolve(unsold, kernel, mode='same')

# print(np.average(regular), np.average(no_harvest), np.average(harvest))

harvest_end = np.argmax(np.cumsum(harvest) > sum(no_harvest))
unsold_end = np.argmax(np.cumsum(unsold) > sum(no_harvest))

if harvest_end == 0:
    harvest_end = -1
if unsold_end == 0:
    unsold_end = -1

print(harvest_end, unsold_end)
# print(harvest_start / (harvest_end - harvest_start), harvest_start / (unsold_end - harvest_start))
# print(np.average(unsold[harvest_start:unsold_end]), np.average(harvest[harvest_start:harvest_end]), np.average(no_harvest[harvest_start:]), np.average(harvest), np.average(regular), np.average(no_harvest))

x_values_static = np.arange(len(no_harvest))
x_values_regular = np.arange(len(regular))
x_values_harvest = np.arange(harvest_end - harvest_start) + harvest_start
x_values_unsold = np.arange(unsold_end - harvest_start) + harvest_start

Figure = plt.figure(figsize=(2.3,1.1))
Graph = Figure.add_subplot(111)
PDF = PdfPages(f"{DIR}/harvest_benefits.pdf")
#color1, color2, color3, color4 = '#EC7497', '#fac205', '#95d0fc', '#96f97b'

plt.plot(x_values_harvest, harvest[harvest_start:harvest_end], '-', label="Sold", color='blue', linewidth=1.5, zorder=3)
# plt.plot(harvest, '-', label="Sold", color='blue', linewidth=1.5, zorder=3)
# plt.plot(no_harvest, '-', label="Static", color='red', linewidth=1.5, zorder=3)
plt.plot(x_values_unsold, unsold[harvest_start:unsold_end], '--',label="Unsold", color='green', linewidth=1.5, zorder=2)
plt.plot(x_values_static, no_harvest, '-',label="Static", color='red', linewidth=1.5, zorder=4)
plt.axvline(x=len(harvest), color='black', linestyle='--', zorder=0)

YLabel = plt.ylabel("Bandwidth (MB/s)", multialignment='center', fontsize=6)
YLabel.set_position((0.0,0.5))
YLabel.set_linespacing(0.5)

# x_labels = ['0','15','30','45','60','75','90','105', '120']
#x_labels = ['0','','30','','60','','90','', '120']
num_labels = 5
Xticks = np.arange(0,num_labels+1)*len(no_harvest)/5
x_labels = [x // 60 for x in Xticks]
Graph.set_xticks(Xticks)
Graph.set_xticklabels(x_labels,fontsize=6)
Graph.xaxis.set_ticks_position('bottom')
Graph.set_xlabel('Time (minutes)', fontsize=6)
Graph.set_xlim((0,len(no_harvest)))

y_labels = ['', '', '200','', '400', '', '600', '', '800'][:7]
YTicks = np.arange(0,7)*100
Graph.set_yticks(YTicks)
Graph.set_yticklabels(y_labels,fontsize=6)
Graph.yaxis.set_ticks_position('none')
Graph.set_ylim((0,650))

Graph.set_axisbelow(True)
Graph.yaxis.grid(color='lightgrey', linestyle='solid')

lg = Graph.legend(loc='upper left', prop={'size':6}, ncol=1, borderaxespad=0.2)
lg.draw_frame(False)

Graph.grid(b=True, which='minor')


plt.margins(0)

PDF.savefig(Figure, bbox_inches='tight')
#plt.savefig("harvest_benefits_terasort.png")
PDF.close()
