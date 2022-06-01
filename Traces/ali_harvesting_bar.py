#!/usr/bin/python3

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from operator import truediv

hr_1 = []
hr_6 = []
hr_12 = []
hr_72 = []

total_vms = 4000

# Here we filter out the trailing zeros to ignore the last few timestamps in the trace. For a 7-day trace and 12 hours of harvesting, the final 12 hours are not considered harvestable. 
with open('ali_harvesting_bar.dat', 'r') as f:
    data = f.readlines()
    for i in range(3):
        hr_1.append(np.mean(list(filter(lambda x: x!= 0,map(float, data[i*4+0].split(","))))))
        hr_6.append(np.mean(list(filter(lambda x: x!= 0,map(float, data[i*4+1].split(","))))))
        hr_12.append(np.mean(list(filter(lambda x: x!= 0,map(float, data[i*4+2].split(","))))))
        hr_72.append(np.mean(list(filter(lambda x: x!= 0,map(float, data[i*4+3].split(","))))))


x_labels = ['10%','25%','50%']
y_labels = ['0', '', '20','', '40', '', '60', '', '80', '', '100']
x_values = np.arange(len(x_labels))

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'
matplotlib.rcParams['hatch.linewidth'] = 0.5

Figure = plt.figure( figsize=(3.0,1) )
Graph = Figure.add_subplot(111)
PDF = PdfPages( "ali_harvesting_bar.pdf" )

BarWidth = 0.15

YLabel = plt.ylabel( '% of VMs', fontsize=7 )
Graph.set_xticks( x_values )
Graph.set_xticklabels( x_labels, fontsize=7, ha="center")
Graph.xaxis.set_ticks_position( 'none' )
Graph.tick_params(pad=0)

YTicks = np.arange(0,1.1,0.1)
Graph.set_yticks( YTicks )
Graph.set_yticklabels(y_labels, fontsize=7 )
Graph.yaxis.set_ticks_position( 'none' )

Graph.set_axisbelow(True)
Graph.yaxis.grid(color = 'lightgray', linestyle= 'solid')

color1, color2, color3, color4 = '#EC7497', '#fac205', '#95d0fc', '#96f97b'

Rects1 = Graph.bar(x_values - BarWidth, hr_1, BarWidth, edgecolor='black', label="1 hr",hatch="--", color=color1)
Rects2 = Graph.bar(x_values, hr_6, BarWidth, edgecolor='black', label="6 hrs", hatch="", color=color2)
Rects3 = Graph.bar(x_values + BarWidth, hr_12, BarWidth, edgecolor='black', label="12 hrs", hatch="////", color=color3)
Rects3 = Graph.bar(x_values + 2*BarWidth, hr_72, BarWidth, edgecolor='black', label="3 days", hatch="\\"*4, color='lightcyan')

Graph.set_xlim( ( BarWidth-1, len(x_values)) )
Graph.set_ylim( ( 0, 1.05 ) )
Graph.set_xlabel('Storage capacity', fontsize=7)

lg=Graph.legend(prop={'size':5}, ncol=4, loc='upper center', borderaxespad=0.)
lg.draw_frame(False)

PDF.savefig( Figure, bbox_inches='tight' )
PDF.close()
