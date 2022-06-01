#!/usr/bin/python3

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
#import statistics
def label_rects(rects):
    """
    Attach a text label above each bar displaying the height
    """
    for rect in rects:
        height = round(rect.get_height(),2)
        Graph.text(rect.get_x() + rect.get_width()/2.0, 1.01 * height, f"{height}", ha='center', va='bottom', fontsize=6,weight='bold')

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'
matplotlib.rcParams['hatch.linewidth'] = 0.5

#Collect data
x_values = []
y_values = []
with open("dur_sz_overprov.dat", 'r') as f:
    data = f.readlines()
    ts_values = list(map(float, data[0].split(",")))
    ml_values = list(map(float, data[1].split(",")))
    pg_values = list(map(float, data[2].split(",")))
    al_values = list(map(float, data[3].split(",")))

o0, o5, o10, o30  = list(zip(ts_values, ml_values, pg_values, al_values))

BarWidth = 0.18

Figure = plt.figure(figsize=(2.3,1.3))
Graph = Figure.add_subplot(111)
PDF = PdfPages("dur_sz_overprov.pdf")

x_labels = ['TeraSort','ML Prep','PageRank', 'Google']
x_values = np.arange(len(x_labels))

color1, color2, color3, color4, color5 = '#EC7497', '#fac205', '#95d0fc', 'lightcyan', 'lightcyan'

#plt.title("Percentage Disk I/O per Container", fontsize=14)

YLabel = plt.ylabel("Accuracy (%)", multialignment='center', fontsize=7)
YLabel.set_position((0.0,0.5))
YLabel.set_linespacing(0.5)

#Want the labels to be days, and the increments are on a 10s basis (hence 360 not 3600)
Xticks = np.arange(len(x_labels))
Graph.set_xticks(Xticks)
#Graph.set_xticklabels(x_labels,fontsize=6, ha="center")
Graph.set_xticklabels(x_labels,fontsize=7,rotation=35)
Graph.tick_params(pad=0)
Graph.xaxis.set_ticks_position('none')
#Graph.set_xlabel('Workload', fontsize=12)

labels = ['0', '', '20','', '40', '50', '60', '70', '80', '90', '100']
YTicks = np.arange(0, 1.1, 0.1)
Graph.set_yticks(YTicks)
Graph.set_yticklabels(labels,fontsize=7)
#Where are the ticks?
Graph.yaxis.set_ticks_position('none')

Graph.set_axisbelow(True)
Graph.yaxis.grid(color='lightgrey', linestyle='solid')

Rects1 = Graph.bar(x_values - 3*BarWidth/2, o0, BarWidth, edgecolor='black', label="None",hatch="\\\\", color=color1)
Rects2 = Graph.bar(x_values - BarWidth/2, o5, BarWidth, edgecolor='black', label="5%", hatch="--", color=color2)
Rects3 = Graph.bar(x_values + BarWidth/2, o10, BarWidth, edgecolor='black', label="10%", hatch="", color=color3)
Rects4 = Graph.bar(x_values + 3*BarWidth/2, o30, BarWidth, edgecolor='black', label="30%", hatch="////", color=color5)

lg = Graph.legend(loc='upper center', prop={'size':6}, ncol=3, borderaxespad=0.0, frameon=False)
#lg.draw_frame(False)

Graph.grid(b=True, which='minor')

Graph.set_xlim( ( BarWidth-1, len(x_values)) )
Graph.set_ylim((0.5,1.09))

PDF.savefig(Figure, bbox_inches='tight')
PDF.close()

#plt.savefig("dur_sz_overprov.png")
#plt.savefig("dur_sz_overprov.pdf")
