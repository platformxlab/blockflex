import os,sys,re

#This shoud process all possible sized values and print results

BASE= "./"
OUT_FILE="/home/js39/ml_res/ml_outputs.txt"
if len(sys.argv) > 1:
    DIR = BASE+sys.argv[1]+"/"
    OUT_FILE=BASE+sys.argv[2]
else:
    print("Need to specify a directory")
    exit()

windows = [180]
buffers = [1, 1.05, 1.1, 1.2, 1.3, 1.4]
wls = ["terasort", "ml_prep", "pagerank", "alibaba"]
test_str = "tmpa"
if sys.argv[1] == 'outputs/sz' or sys.argv[1] == 'outputs/dur_sz':
    wls[3] = "google"
    test_str = "tmpg"


#Gather results
over =    [[[] for _ in range(len(buffers))] for _ in range(len(wls))] 
under =   [[[] for _ in range(len(buffers))] for _ in range(len(wls))]
correct = [[[] for _ in range(len(buffers))] for _ in range(len(wls))]
comb_correct = [[[] for _ in range(len(buffers))] for _ in range(len(wls))]



for filename in os.listdir(DIR):
    w = int(test_str.index(filename[0]))
    #print(filename)
    i,_ = map(int, re.findall(r'\d+', filename))

    with open(DIR+filename,'r') as f:
        for line in f:
            if line.startswith("overpred"):
                _,o,_,c,_,u = line.replace(',', "").strip().split(" ")
                over[w][i].append(float(o))
                under[w][i].append(float(u))
                correct[w][i].append(float(c))
                if w < 3 or sys.argv[1][0] != 'd':
                    comb_correct[w][i].append(float(c) + float(o))
                else:
                    comb_correct[w][i].append(float(c) + float(u))

#Print results
with open(OUT_FILE, 'w') as f:
    for i, wl in enumerate(wls):
        print(f"====={wl}=====", file=f)
        for j, buf in enumerate(buffers):
            for k, window in enumerate(windows):
                print(f"\t=====window={window}=buffer={buf}", file=f)
                cur_over = over[i][j]
                cur_under = under[i][j]
                cur_correct = correct[i][j]
                cur_comb = comb_correct[i][j]
                if not cur_over or not cur_under or not cur_correct or not cur_comb: continue
                print(f"\t\tUNDER: max: {max(cur_under)} min: {min(cur_under)} avg: {sum(cur_under)/len(cur_under)}", file=f)
                print(f"\t\tCORRECT: max: {max(cur_correct)} min: {min(cur_correct)} avg: {sum(cur_correct)/len(cur_correct)}", file=f)
                print(f"\t\tOVER: max: {max(cur_over)} min: {min(cur_over)} avg: {sum(cur_over)/len(cur_over)}", file=f)
                print(f"\t\tCOMB CORRECT: max: {max(cur_comb)} min: {min(cur_comb)} avg: {sum(cur_comb)/len(cur_comb)}", file=f)

