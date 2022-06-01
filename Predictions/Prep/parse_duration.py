import sys
import statistics
import numpy as np
import math

infile = sys.argv[1]
outfile = sys.argv[2]
in_file = open(infile, "r")
window = 180
sub_window = 30
out_file = open(outfile, "w")
if len(sys.argv) >= 4:
    window = int(sys.argv[3])

MB = 1000000
bw = 64 * MB
read_size = 0
write_size = 0
read_op = 0
write_op = 0
sub_write = []
sub_read = []
sub_total = []
sub_writeop = []
sub_readop = []
sub_totalop = []
thresh = -1.0
t_sub = -1.0
for line in in_file:
    #Split the input line
    temp = line.split()

    #Bad line checks 
    if len(temp) < 8:
        continue
    try:
        ts = float(temp[3])
    except ValueError:
        continue

    #Just to avoid weirdness for weird first measurements
    if thresh < 0:
        thresh = ts + window
        t_sub = ts + sub_window
    size = int(temp[7])
    #Read operation
    if temp[6].startswith("R"):
        read_size += size
        read_op += 1
    #Write operation
    if temp[6].startswith("W"):
        write_size += size
        write_op += 1
    #We have completed on "sub" interaval
    if float(ts) > t_sub:
        #Reset next sub-interval threshold
        #while float(ts) > t_sub:
        t_sub += sub_window
        #Record subintervals for later
        read_size /= sub_window
        write_size /= sub_window
        write_op /= sub_window
        read_op /= sub_window
        sub_total.append(read_size + write_size)
        sub_read.append(read_size)
        sub_write.append(write_size)
        sub_readop.append(read_op)
        sub_writeop.append(write_op)
        sub_totalop.append(read_op + write_op)
        #Reset stats for next sub-interval
        read_size = 0
        write_size = 0
        write_op = 0
        read_op = 0

    #We have completed one full interval, can aggregate across the sub intervals
    if float(ts) > thresh:
        #Reset next threshold
        #while float(ts) > thresh:
        thresh += window
        print(f"{window} {min(int(math.ceil(max(sub_total)))//bw, 16)}", file=out_file)
        sub_read = []
        sub_write = []
        sub_total = []
        sub_readop = []
        sub_writeop = []
        sub_totalop = []

in_file.close()
out_file.close()
