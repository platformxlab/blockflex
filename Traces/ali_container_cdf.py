#!/bin/python3

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

total_vms = 4000
start_time = 10000
with open("ali_harvesting_bar.dat", "w") as out_f, open('ali_stats_util_cdf.txt', 'r') as f:
    data = f.readlines()

    for i in range(3):
        one_hour_harvesting = np.array(list(map(float, eval(data[i*10+1]).values()))) / total_vms
        six_hours_harvesting = np.array(list(map(float, eval(data[i*10+3]).values()))) / total_vms
        twelve_hours_harvesting = np.array(list(map(float, eval(data[i*10+5]).values()))) / total_vms
        three_days_harvesting = np.array(list(map(float, eval(data[i*10+9]).values()))) / total_vms

        ys = [one_hour_harvesting, six_hours_harvesting, twelve_hours_harvesting, three_days_harvesting]
        for y in ys:
            out_f.write(",".join([str(_) for _ in y[start_time:]])+"\n")

