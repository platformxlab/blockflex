#!/usr/bin/python3

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os

plt.rcParams.update({'font.family': 'serif'})

avg = []
peak = []
mins = []
with open("google_util_cdf.dat",'r') as f:
    for line in f:
        #length 6, want the last two
        temp = line.strip().split(" ")
        min_util = float(temp[6])
        peak_util = float(temp[5])
        avg_util = float(temp[4])
        if peak_util < avg_util or peak_util > 1 or avg_util > 1 or min_util > avg_util or min_util <0:
            continue
        avg.append(avg_util)
        peak.append(peak_util)
        mins.append(min_util)

mins = np.array(mins)
peak = np.array(peak)
avg = np.array(avg)
peak = np.sort(peak)
avg = np.sort(avg)
mins = np.sort(mins)

y_values = np.arange(len(avg))/len(avg) + 1.0/len(avg)
y_values = np.flip(y_values)


Figure = plt.figure(figsize=(2.8,0.9))
Graph = Figure.add_subplot(111)
PDF = PdfPages("google_util_cdf.pdf")

plt.plot(y_values, peak, '-', label="Maximum",color='red',linewidth=2)
plt.plot(y_values, avg, '--', label="Average",color='blue',linewidth=2)
plt.plot(y_values, mins, ':' ,label="Minimum",color='green',linewidth=2)

labels = ['0', '', '20','', '40', '', '60', '', '80', '', '100']

YLabel = plt.ylabel("Storage Capacity \n\n Utilization (%)", multialignment='center', fontsize=7)
YLabel.set_position((0.0,0.5))
YLabel.set_linespacing(0.5)

#Want the labels to be days, and the increments are on a 10s basis (hence 360 not 3600)
Xticks = np.arange(0, 1.1, 0.1)
Graph.set_xticks(Xticks)
Graph.set_xticklabels(labels,fontsize=7)
Graph.xaxis.set_ticks_position('none')
Graph.set_xlabel('Percentage of VMs (%)', fontsize=7)

YTicks = Xticks
Graph.set_yticks(YTicks)
Graph.set_yticklabels(labels,fontsize=7)
Graph.yaxis.set_ticks_position('none')

Graph.set_axisbelow(True)
Graph.yaxis.grid(color='lightgrey', linestyle='solid')

lg = Graph.legend(loc='upper right', prop={'size':7}, ncol=1, borderaxespad=0.2)
lg.draw_frame(False)

Graph.grid(b=True, which='both')

Graph.set_xlim((0,1))
Graph.set_ylim((0,1))
plt.margins(0)

PDF.savefig(Figure, bbox_inches='tight')
PDF.close()
