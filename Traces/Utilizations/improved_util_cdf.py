
#!/usr/bin/python3

import matplotlib
from numpy.core.fromnumeric import mean
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import sys

trace = sys.argv[1]

assert(trace in ['alibaba', 'google'])

plt.rcParams.update({'font.family': 'serif'})

file = f"{trace}_improved_util.dat"
file1 = "original_util.txt"
file2 = "improved_util.txt"
file3 = "harvest_min_max.txt"

original = dict()
improved = dict()
improved_max = dict()
min_util = dict()
max_util = dict()

with open(file1) as f1:
    for i, line in enumerate(f1):
        _id, util, _ = line.strip().split(" ")
        original[_id] = float(util)

with open(file2) as f2:
    for i, line in enumerate(f2):
        _id, util, _max_util = line.strip().split(" ")
        improved[_id] = float(util)
        improved_max[_id] = float(_max_util)

with open(file3) as f3:
    for i, line in enumerate(f3):
        _id, max, avg, min = line.strip().split(",")[:4]
        if _id in improved:
            min_util[_id] = 1 - float(min)
            max_util[_id] = 1 - float(max)

with open(file, "w") as outfile:
    for _id in improved:
        outfile.write(f"{_id} {original[_id]} {improved[_id]} {min_util[_id]} {max_util[_id]} {improved_max[_id]}\n")
