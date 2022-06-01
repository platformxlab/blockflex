import numpy as np
from macros import *

scheduling_interval = 10

def convert_trace(filename):
    traces = []
    bandwidth = defaultdict(int)
    log_msg("Start parsing")
    with open(f"{BLOCKFLEX_DIR}/raw_traces/{filename}_offset.trace") as open_f:
        for i, line in enumerate(open_f):
            if i != 0 and i % 100000 == 0:
                log_msg(f"Start parsing {i}")
            
            raw = [_ for _ in line.strip().split(" ") if _ != ""]
            
            if len(raw) < 10:
                continue
            if raw[6][0] == "R":
                mode = 0
            else:
                mode = 1

            ts, mode, off, size = float(raw[3]), mode, int(raw[7]) * 512, int(raw[9]) * 512
            traces.append((ts, mode, off, size))
            bandwidth[int(ts // scheduling_interval)] += size / scheduling_interval / 1024**2

    traces = np.array(traces)
    np.save(f"traces/{filename}.npy", traces)

    max_t = max(list(bandwidth.keys()))
    bandwidth_record = []
    for t in range(max_t+1):
        bandwidth_record.append(bandwidth[t])
    bandwidth_record = np.array(bandwidth_record)
    np.save(f"traces/{filename}_bw.npy", bandwidth_record)

    return traces

# convert_trace("terasort")
# convert_trace("pagerank")
# convert_trace("ml_prep")
# convert_trace("ycsb")

traces = np.load(f"traces/ml_prep_bw.npy")
print(traces, len(traces))