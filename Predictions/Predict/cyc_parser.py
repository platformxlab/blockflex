import os,sys,re

#This shoud process all possible sized values and print results

BASE= "/home/js39/ml_res/js_outputs/"
OUT_FILE="/home/js39/ml_res/ml_cyc_outputs.txt"
if len(sys.argv) > 1:
    DIR = BASE+sys.argv[1]+"/"
    OUT_FILE=BASE+sys.argv[2]
else:
    print("Need to specify a directory")
    exit()

#TODO probably restrict this
windows = [180]
#windows = [60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
buffers = [1, 1.05, 1.1, 1.2, 1.5, 2]
wls = ["terasort", "ml_prep", "pagerank", "alibaba"]
test_str = "tmpa"
if sys.argv[1] == 'sz' or sys.argv[1] == 'dur_sz':
    wls[3] = "google"
    test_str = "tmpg"

core_freq = 3_000_000_000

#Gather results, note here sizing by buffers, not windows
train_samps = [[[] for _ in range(len(buffers))] for _ in range(len(wls))]
train_cycles = [[[] for _ in range(len(buffers))] for _ in range(len(wls))]
train_per = [[[] for _ in range(len(buffers))] for _ in range(len(wls))]
pred_samps = [[[] for _ in range(len(buffers))] for _ in range(len(wls))]
pred_cycles = [[[] for _ in range(len(buffers))] for _ in range(len(wls))]
pred_per = [[[] for _ in range(len(buffers))] for _ in range(len(wls))]


for filename in os.listdir(DIR):
    w = int(test_str.index(filename[0]))
    i,_ = map(int, re.findall(r'\d+', filename))
    with open(DIR+filename,'r') as f:
        for line in f:
            if line.startswith("Training Overhead"):
                _,_,_,s,_,c,_,p = line.strip().split(" ")
                train_samps[w][i].append(float(s))
                train_cycles[w][i].append(float(c))
                train_per[w][i].append(float(p))
            if line.startswith("Prediction Overhead"):
                _,_,_,s,_,c,_,p = line.strip().split(" ")
                pred_samps[w][i].append(float(s))
                pred_cycles[w][i].append(float(c))
                pred_per[w][i].append(float(p))

#Print results
with open(OUT_FILE, 'w') as f:
    for i, wl in enumerate(wls):
        print(f"====={wl}=====", file=f)
        for j, buf in enumerate(buffers):
            for k, window in enumerate(windows):
                print(f"\t=====window={window}=buffer={buf}", file=f)
                cur_train = train_per[i][j]
                cur_pred = pred_per[i][j]
                if not cur_train or not cur_pred: continue
                print(f"\t\tTRAIN CYCLES: max: {max(cur_train)} min: {min(cur_train)} avg: {sum(cur_train)/len(cur_train)}", file=f)
                print(f"\t\tTRAIN SECONDS: max: {max(cur_train)/core_freq} min: {min(cur_train)/core_freq} avg: {sum(cur_train)/len(cur_train)/core_freq}", file=f)
                print(f"\t\tPRED CYCLES: max: {max(cur_pred)} min: {min(cur_pred)} avg: {sum(cur_pred)/len(cur_pred)}", file=f)
                print(f"\t\tPRED SECONDS: max: {max(cur_pred)/core_freq} min: {min(cur_pred)/core_freq} avg: {sum(cur_pred)/len(cur_pred)/core_freq}", file=f)

