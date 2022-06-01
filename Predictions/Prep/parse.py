import sys
import statistics
import numpy as np

infile = sys.argv[1]
outfile = sys.argv[2]
in_file = open(infile, "r")
window = 180.0
if len(sys.argv) == 4:
    window = float(sys.argv[3])

sub_window = 30.0
out_file = open(outfile, "w")

def parse(sub_r, sub_w, sub_rop, sub_wop, sub_tot, sub_totop):
    #Get max
    max_r_bw = max(sub_r) if len(sub_r) > 0 else 0.0
    max_w_bw = max(sub_w) if len(sub_w) > 0 else 0.0
    max_tot_bw = max(sub_tot) if len(sub_tot) > 0 else 0.0
    max_r_op = max(sub_rop) if len(sub_rop) > 0 else 0.0
    max_w_op = max(sub_wop) if len(sub_wop) > 0 else 0.0
    max_tot_op = max(sub_totop) if len(sub_totop) > 0 else 0.0
    #Get min
    min_r_bw = min(sub_r) if len(sub_r) > 0 else 0.0
    min_w_bw = min(sub_w) if len(sub_w) > 0 else 0.0
    min_tot_bw = min(sub_tot) if len(sub_tot) > 0 else 0.0
    min_r_op = min(sub_rop) if len(sub_rop) > 0 else 0.0
    min_w_op = min(sub_wop) if len(sub_wop) > 0 else 0.0
    min_tot_op = min(sub_totop) if len(sub_totop) > 0 else 0.0
    #Get average
    avg_r_bw = sum(sub_r)/len(sub_r) if len(sub_r) > 0 else 0.0
    avg_w_bw = sum(sub_w)/len(sub_w) if len(sub_w) > 0 else 0.0
    avg_tot_bw = sum(sub_tot)/len(sub_tot) if len(sub_tot) > 0 else 0.0
    avg_r_op = sum(sub_rop)/len(sub_rop) if len(sub_rop) > 0 else 0.0
    avg_w_op = sum(sub_wop)/len(sub_wop) if len(sub_wop) > 0 else 0.0
    avg_tot_op = sum(sub_totop)/len(sub_totop) if len(sub_totop) > 0 else 0.0
    #Get median
    med_r_bw = statistics.median(sub_r) if len(sub_r) > 0 else 0.0
    med_w_bw = statistics.median(sub_w) if len(sub_w) > 0 else 0.0
    med_tot_bw = statistics.median(sub_tot) if len(sub_tot) > 0 else 0.0
    med_r_op = statistics.median(sub_rop) if len(sub_rop) > 0 else 0.0
    med_w_op = statistics.median(sub_wop) if len(sub_wop) > 0 else 0.0
    med_tot_op = statistics.median(sub_totop) if len(sub_totop) > 0 else 0.0
    #Get stddev
    std_r_bw = statistics.stdev(sub_r) if len(sub_r) > 1 else 0.0
    std_w_bw = statistics.stdev(sub_w) if len(sub_w) > 1 else 0.0
    std_tot_bw = statistics.stdev(sub_tot) if len(sub_tot) > 1 else 0.0
    std_r_op = statistics.stdev(sub_rop) if len(sub_rop) > 1 else 0.0
    std_w_op = statistics.stdev(sub_wop) if len(sub_wop) > 1 else 0.0
    std_tot_op = statistics.stdev(sub_totop) if len(sub_totop) > 1 else 0.0
    #This is currently redundant
    nparr = np.array([max_r_bw, max_w_bw, max_tot_bw, max_r_op, max_w_op, max_tot_op, min_r_bw, min_w_bw, min_tot_bw, min_r_op, min_w_op, min_tot_op, avg_r_bw, avg_w_bw, avg_tot_bw, avg_r_op, avg_w_op, avg_tot_op, med_r_bw, med_w_bw, med_tot_bw, med_r_op, med_w_op, med_tot_op, std_r_bw, std_w_bw, std_tot_bw, std_r_op, std_w_op, std_tot_op])
    #Normalize to window length
    return nparr.tolist()



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
#print("[max_r_bw, max_w_bw, max_tot_bw, max_r_op, max_w_op, max_tot_op, min_r_bw, min_w_bw, min_tot_bw, min_r_op, min_w_op, min_tot_op, avg_r_bw, avg_w_bw, avg_tot_bw, avg_r_op, avg_w_op, avg_tot_op, sum_r_bw, sum_w_bw, sum_tot_bw, sum_r_op, sum_w_op, sum_tot_op, med_r_bw, med_w_bw, med_tot_bw, med_r_op, med_w_op, med_tot_op, std_r_bw, std_w_bw, std_tot_bw, std_r_op, std_w_op, std_tot_op") 
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
        stats = parse(sub_read, sub_write, sub_readop, sub_writeop, sub_total, sub_totalop)
        print(','.join(map(str,stats)), file=out_file)
        sub_read = []
        sub_write = []
        sub_total = []
        sub_readop = []
        sub_writeop = []
        sub_totalop = []
if read_op + write_op != 0:
    stats = parse(sub_read, sub_write, sub_readop, sub_writeop, sub_total, sub_totalop)
    print(",".join(map(str,stats)), file=out_file)

in_file.close()
out_file.close()
