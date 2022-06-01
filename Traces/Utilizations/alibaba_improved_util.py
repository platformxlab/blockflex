
#!/usr/bin/python3

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os


plt.rcParams.update({'font.family': 'serif'})

file = "alibaba_improved_util.dat"

original_util = dict()
improved_util = dict()
improved_max_util = dict()
min_util = dict()
max_util = dict()


with open(file) as f:
    for i, line in enumerate(f):
        _id, orig, improved, min, max, improved_max = line.strip().split(" ")
        improved_util[_id] = float(improved)
        improved_max_util[_id] = float(improved_max)
        original_util[_id] = float(orig)
        min_util[_id] = float(min)
        max_util[_id] = float(max)



sum_orig = 0
sum_improved = 0
for _id in original_util:
    if original_util[_id] <= 0.6:
        sum_improved += improved_util[_id] 
        sum_orig += original_util[_id]

print("Improved util for VM < 60% util", sum_improved / sum_orig)

print(sum(improved_util.values()) / len(original_util), sum(original_util.values()) / len(original_util))

print("Overall improvement", (sum(improved_util.values()) / len(original_util)) / (sum(original_util.values()) / len(original_util)))

print("Overall improvement", (sum(improved_max_util.values()) / len(original_util)) , (sum(max_util.values()) / len(original_util)))

original_util = sorted(original_util.values(), reverse=True)
improved_util = sorted(improved_util.values(), reverse=True)
improved_max_util = sorted(improved_max_util.values(), reverse=True)

min_util = sorted(min_util.values(), reverse=True)
max_util = sorted(max_util.values(), reverse=True)

x_values = np.arange(len(original_util))/len(original_util) 

#BarWidth = 0.55

Figure = plt.figure(figsize=(2.8,1.0))
Graph = Figure.add_subplot(111)
PDF = PdfPages("alibaba_improved_util_cdf.pdf")

plt.plot(x_values, improved_util, '-.', label="Improved Avg",color='m',linewidth=2)
plt.plot(x_values, original_util, '--', label="Average",color='blue',linewidth=2)
plt.plot(x_values, min_util, ':', label="Min",color='green',linewidth=2)
plt.plot(x_values, max_util, '-', label="Max",color='red',linewidth=2)

plt.plot(x_values, improved_max_util, '-', label="Improved Max",color='black',linewidth=2)
#plt.title("Percentage Disk I/O per Machine", fontsize=14)

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
#What are the action labels?
Graph.set_yticklabels(labels,fontsize=7)
#Where are the ticks?
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
#plt.savefig("alibaba_util_cdf.png")
PDF.close()
