import time
from joblib import Parallel, delayed
# import multiprocessing
from collections import defaultdict
import os
from macros import *

scheduling_interval = 10
bw_ssd = 700
np.set_printoptions(precision=2)
def open_trace(filename):
    TRACE_FILE = f"{USR_DIR}/traces/{filename}.npy"
    TRACE_BW_FILE = f"{USR_DIR}/traces/{filename}_bw.npy"
    traces = np.load(TRACE_FILE)
    bw_traces = np.load(TRACE_BW_FILE)
    
    log_msg("Finished parsing", out_f=out_f)

    return traces, bw_traces

def perform_disk_io(mode, disk, trace, bw_ref):
    off, size = int(trace[2]), int(trace[3])
    
    computation_delay = size / bw_ref / MB - size / bw_ssd / MB
    if computation_delay > 0:
        time.sleep(computation_delay)

    try:
        if mode == 0:
            disk.seek(off)
            data = disk.read(size)
        elif mode == 1:
            disk.seek(off)
            disk.write(bytearray(size))
    except OSError:
        print(f"Skip request {trace[0]} {mode} {off} {size}")


def run_trace(disk_id, vm_type, traces, bw_traces, tid=0, total_threads=1):
    start = time.time()
    writes, write_size = 0, 0
    reads, read_size = 0, 0
    total_s = 0
    total_r = 0
    PAGE_SIZE = 16384
    assert(disk_id not in ['/dev/sda', '/dev/sdb', '/dev/sdc', '/dev/sdd'])
    disk = open(disk_id,'r+b')
    
    log_msg("Start replaying", vm_type, out_f=out_f)
    while True:
        aggregated_read = None
        for i, t in enumerate(traces):
            if i % total_threads != tid:
                continue
            total_r += 1
            bw_ref = bw_traces[int(t[0] // scheduling_interval)] 
            if total_r % 1000000 == 0:
                log_msg(f"Start processing {total_r}", out_f=out_f)
                end = time.time()
                log_msg(disk_id, writes, write_size, reads, read_size, end-start, total_s, out_f=out_f)
            if t[1] == 0:
                reads += 1
                read_size += t[3]
                if aggregated_read is None:
                    aggregated_read = [t[0], 0, t[2], t[3]]
                else:
                    if aggregated_read[2] + aggregated_read[3] != t[2]:
                        # flush previous read
                        perform_disk_io(0, disk, aggregated_read, bw_ref)
                        aggregated_read = [t[0], 0, t[2], t[3]]
                    else:
                        aggregated_read[3] += t[3]
                
                if aggregated_read[3] >= PAGE_SIZE:
                    perform_disk_io(0, disk, aggregated_read, bw_ref)
                    aggregated_read = None

            elif t[1] == 1:
                writes += 1
                write_size += t[3]
                perform_disk_io(1, disk, t, bw_ref)
        

    disk.close()
    end = time.time()

    log_msg(disk_id, writes, write_size, reads, read_size, end-start, out_f=out_f)

trace = sys.argv[1]
out_f = None
if len(sys.argv) >= 3:
    outfile = sys.argv[2]
    out_f = open(outfile, "w+")

if trace in ["terasort", "ml_prep", "pagerank"]:
    DISK_ID = '/dev/sde'
    TYPE = 'harvest'
    NUM_THREADS = 1
elif trace == "ycsb":
    DISK_ID = '/dev/sdf'
    TYPE = 'regular'
    NUM_THREADS = 1
elif trace == "test":
    trace = 'terasort'
    DISK_ID = 'dummy_dev'
    TYPE = 'harvest'
    NUM_THREADS = 1

traces, bw_traces = open_trace(trace)

if NUM_THREADS == 1:
    run_trace(DISK_ID, TYPE, traces, bw_traces)
else:
    Parallel(n_jobs=NUM_THREADS)(delayed(run_trace)(DISK_ID, TYPE, traces, bw_traces, tid=i, total_threads=NUM_THREADS) for i in range(NUM_THREADS))
