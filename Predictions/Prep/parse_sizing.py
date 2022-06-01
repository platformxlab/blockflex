import sys
import statistics
import numpy as np
import math

infile = sys.argv[1]
outfile = sys.argv[2]
in_file = open(infile, "r")
window = 180.0
sub_window = 30.0
dur_window = 180
out_file = open(outfile, "w")
df = outfile+"_dur"
if len(sys.argv) == 5:
    dur_window = float(sys.argv[3])
    df = sys.argv[4]

dur_file = open(df, "w")

def parse(diffs):
    #Get max
    maxx = max(diffs) if len(diffs) > 0 else 0.0
    #Get min
    minn = min(diffs) if len(diffs) > 0 else 0.0
    #Get average
    avg = sum(diffs)/len(diffs) if len(diffs) > 0 else 0.0
    #Get median
    med = statistics.median(diffs) if len(diffs) > 0 else 0.0
    #Get stddev
    std = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    #This is currently redundant
    nparr = np.array([maxx, minn, avg, med, std])
    #Normalize to window length
    return nparr.tolist()

read_size = 0
write_size = 0
read_op = 0
write_op = 0
diff = []
thresh = -1.0
GB = 1000000
incs = 32
hist = []
window_size = window//sub_window
cur_window = 0
dur_hist = []
for line in in_file:

    #Split the input line
    temp = line.split()

    size = int(temp[0])
    if thresh < 0:
        thresh = size
        t_sub = size

    cur_window += 1
    hist.append(size)
    dur_hist.append(size)
    if cur_window != sub_window:
        continue
    cur_window = 0
    #Just to avoid weirdness for weird first measurements
    diff.append((size - t_sub) * window_size /GB)
    t_sub = size
    #We have completed on "sub" interaval
    if len(diff) >= window_size:
        #Reset next sub-interval threshold
        #while float(ts) > t_sub:
        #Record subintervals for later
        stats = parse(diff)
        val = math.ceil(max(hist)//GB/incs)
        stats.append(val)
        if len(dur_hist) >= dur_window:
            val = math.ceil(max(dur_hist)//GB/incs)
            print(f"{dur_window} {val}", file=dur_file)
            dur_hist = []
        print(','.join(map(str,stats)), file=out_file)
        hist = []
        diff = []

if len(diff) != 0:
    stats = parse(diff)
    val = math.ceil(max(hist)//GB/incs)
    stats.append(val)
    val = math.ceil(max(dur_hist)//GB/incs)
    print(f"{dur_window} {val}", file=dur_file)
    print(",".join(map(str,stats)), file=out_file)

dur_file.close()
in_file.close()
out_file.close()
