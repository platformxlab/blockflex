import torch.nn as nn
from hwcounter import count, count_end
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import sklearn

windows = [180]

# Here we now use the sequence of last durations we have hit (5 min increments to predict how much
# longer it will endure. This will require somewhat updating the training procedure to when
# we actually have a shift.

DEVICE = torch.device("cpu")

in_file = "c_10005.txt"

if len(sys.argv) >= 2:
    in_file = sys.argv[1]

print("Running with " + in_file)
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=30, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.input_size = input_size
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.softmax = nn.Softmax(dim=2)
        #print(self)

    def init_hidden(self, batch_size):
        self.hidden_cell = (torch.zeros(1,batch_size,self.hidden_layer_size).to(DEVICE),
                            torch.zeros(1,batch_size,self.hidden_layer_size).to(DEVICE))

    def forward(self, inputs):
        features = self.input_size
        batch_size = inputs.shape[0]
        seq_length = inputs.shape[1]
        self.init_hidden(batch_size)
        lstm_out, self.hidden_cell = self.lstm(inputs.view(seq_length, batch_size, features), self.hidden_cell)
        predictions = self.linear(lstm_out.view(seq_length, batch_size, self.hidden_layer_size))
        return predictions[-1]



def train_online(model, optimizer, train_sequence, train_bool, batch_size=1, num=1):
    num_batches = len(train_sequence)//batch_size
    print(f"Total batches {num_batches}")
    pred = []
    variation =0
    correct = 0
    global correct_g
    global under_g
    global over_g
    global total_g
    global pred_time
    global pred_samples
    global train_time
    global train_samples
    o_pred_sum = 0
    u_pred_sum = 0
    above = 0
    below = 0
    total = 0
    prev_train = 0
    n_updates = 0
    train_cnt = 0
    for j in range(num_batches):
        if train_bool[j]: 
            while prev_train <= j:
                n_updates += 1
                batch_seq = train_sequence[prev_train*batch_size:(prev_train+1)*batch_size]
                #split the inputs and labels
                list_inputs, list_labels = list(zip(*batch_seq))
                inputs = torch.stack(list_inputs)
                labels = torch.stack(list_labels)
                model.train()
                start = count()
                optimizer.zero_grad()
                #This seems to be needed in order to make the dimensions match, its just a dummy dimension
                out= torch.unsqueeze(model(inputs), 0)
                single_loss = loss_function(out, labels)
                single_loss.backward()
                optimizer.step()
                train_time += count_end() - start
                prev_train += 1
                train_samples += 1
                train_cnt += 1

        #Predict for the next timestep
        model.eval()
        with torch.no_grad():
            if j !=num_batches-1:
                batch_seq = train_sequence[(j+1)*batch_size:(j+2)*batch_size]
                list_inputs, list_labels = list(zip(*batch_seq))
                inputs = torch.stack(list_inputs)
                labels = torch.stack(list_labels)
                start = count()
                tag_score = model(inputs)
                pred_time += count_end() - start
                if tag_score[0][0] < 0: 
                    tag_score[0][0] = 0
                pred_samples += 1
                label_score = labels
                total += 1
                tag_score_int = int(tag_score*outputs)
                label_score_int = int(label_score*outputs)
                buf_score_int = int(math.floor(tag_score_int * buf))
                if buf_score_int > outputs:
                    buf_score_int = outputs
                print(f"Pred: {tag_score_int}, buffer: {buf_score_int}, label: {label_score_int}")
                if label_score_int < buf_score_int:
                    above += 1
                    o_pred_sum += buf_score_int - label_score_int
                elif label_score_int > tag_score_int:
                    u_pred_sum += label_score_int - tag_score_int
                    below += 1
                else:
                    correct += 1
                #pred.append(tag_score.cpu().detach().numpy()[0])
                variation += abs(tag_score - labels)

    print("Trained with: " + str(train_cnt))
    if num in seen and train_cnt > 5:
        over_g += above
        under_g += below
        correct_g += correct
        total_g += total
        print(f"avg_overpred: {o_pred_sum} {above}")
        print(f"avg_underpred: {u_pred_sum} {below}")
    if total == 0:
        print("Not enough to train with for this size")
    else:
        print(f"temp overpred: {above/total}, correct: {(correct)/total}, underpred: {below/total}")
        #print(f"time per: {(end-start)/(epochs*num_batches)}")
    return pred

def preprocessing(input_data, label_data, win):
    inout_seq = []
    L = len(input_data)
    for i in range(L-win):
        #sends in the input bandwidths for the window length
        train_seq = torch.FloatTensor(input_data[i:i+win]).to(DEVICE)
        train_label= torch.FloatTensor(label_data[i+win:i+win+1]).to(DEVICE)
        if (train_label[0] > outputs): train_label[0] = outputs
        if (train_label[0] < 0): train_label[0] = 0
        train_label[0]/=outputs
        inout_seq.append((train_seq, train_label))
    return inout_seq



def read_input(in_file):
    #Input format:
    #Duration, #channels (by my definition of the trace parsing)
    #Channels here is not an output, but instead is a input value that can help predict
    inp_data = [[] for _ in range(channels)]
    #Open the input data file and read everything in
    cnt = -1
    with open(in_file, "r") as f:
        for line in f:
            cnt += 1
            if cnt == 0:
                continue
            temp = line.split()
            #Need to convert the input time to the nearest half hour
            duration = int(float(temp[0]))
            curbw = int(float(temp[1]))
            seen.add(curbw-1)
            #Setting up the training for ensuring we have 'i' bandwidth or less
            #This each time we have curbw bandwidth we need to train for all bw < curbw
            #The rest will have their durations extended.
            for i in range(channels):
                if len(inp_data[i]) > train_lim:
                    continue
                #Want to train for those with lower
                if i < curbw:
                    #If have hist to train with
                    if hist[i] > 0:
                        duration = hist[i]
                        hist[i] = 0
                        max_hist[i] = 0
                        cur = min(rem, duration)
                        while cur <= duration:
                            inp_data[i].append([cur//time_alloc])
                            #Shouldn't need to care that we are intermixing since each one is generated seperately
                            train_bool[i].append(True)
                            label_data[i].append([(duration - cur)//time_alloc])
                            #Every interval
                            cur += rem
                else:
                    hist[i] += duration
                    #Shave off time until we are under 12 hours since this is the max allocation anyways, might as well train with such
                    while hist[i]-max_hist[i] > out_lim:
                        inp_data[i].append([max_hist[i]/time_alloc])
                        train_bool[i].append(True)
                        label_data[i].append([(hist[i]-max_hist[i])//time_alloc])
                        #Every three minutes
                        max_hist[i] += rem
                    if max_hist[i] == 0 and hist[i] >= time_alloc:
                        temp_train_val = 0
                        best_train_val = hist[i] // time_alloc
                        while temp_train_val + best_train_val * time_alloc <= hist[i]:
                            inp_data[i].append([temp_train_val/time_alloc])
                            train_bool[i].append(True)
                            label_data[i].append([(best_train_val)//time_alloc])
                            temp_train_val += rem
    return inp_data



#History length used for predictions
window = 6
#Not used
batch_size=1 
#24 hours in 30 minute increments
over_g = 0
correct_g = 0
under_g = 0
total_g = 0 
outputs = 25
train_time = 0
train_samples = 0
train_lim = 1500
pred_time = 0
pred_samples = 0
seen = set()

cyc_overhead = 350

rem = 180
buf = 1
buffers = [1/1,1/1.05,1/1.1,1/1.2,1/1.3,1/1.4]
print(buffers)
if len(sys.argv) >= 3:
    buf = buffers[int(sys.argv[2])]

#Smallest allocation is 30 minutes
time_alloc = 1800
#The largest possible allocation in secodns
out_lim =outputs * time_alloc
#How many bandwidth allocations does the ssd have
channels = 17
#What is the current history, used for generating inputs for the training from the trace
hist = [0 for _ in range(channels)]
max_hist = [0 for _ in range(channels)]
#whether the current value is going to be used for training or just for testing accuracy (for now)
train_bool = [[] for _ in range(channels)]
label_data = [[] for _ in range(channels)]

#Set of models for each bandwidth allocation size
model = []
optimizer = []
for i in range(channels):
    model.append(LSTM(input_size=1, output_size=1).to(DEVICE))
    optimizer.append(torch.optim.Adam(model[i].parameters(), lr=0.003))
loss_function = nn.MSELoss()

##DONE SETUP, NOW RUNNING

#Pull the sequence of total/r/w bws from the stats file
input_data = read_input(in_file)

#Take the input and convert into data/label combos
input_seq = []
for i in range(channels):
#for i in range(5):
    input_seq.append(preprocessing(input_data[i], label_data[i], window))

num_samples = len(input_seq)
print(f"num_samples: {num_samples}")

pred_online=[]
for i in range(channels):
    print(f"Training: {i}")
    pred_online = train_online(model[i], optimizer[i], input_seq[i], train_bool[i], batch_size=batch_size,num=i)

if total_g != 0:
    print(f"overpred: {over_g/total_g}, correct: {correct_g/total_g}, underpred: {under_g/total_g}")
    #Adjust for cycle overhead of the tracking itself
    train_over_adjusted = train_time - (train_samples * cyc_overhead)
    train_over_adjusted/=train_samples
    pred_over_adjusted = pred_time - (pred_samples * cyc_overhead)
    pred_over_adjusted/=pred_samples
    print(f"Training Overhead: Samples: {train_samples} Time: {train_time} Per: {train_over_adjusted}")
    print(f"Prediction Overhead: Samples: {pred_samples} Time: {pred_time} Per: {pred_over_adjusted}")

for i in range(channels):
    checkpoint = {'model':model[i], 
            'state_dict': model[i].state_dict(),
            'optimizer':optimizer[i].state_dict()}
    torch.save(checkpoint, str(i)+'_gl_dur.model')
