import torch.nn as nn
from hwcounter import count, count_end
import torch
import numpy as np
import sys
import math

DEVICE = torch.device("cpu")

in_file = "terasort_sizing"

if len(sys.argv) >= 2:
    in_file = sys.argv[1]
print("Running with " + in_file)
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=4, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.input_size = input_size
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.softmax = nn.Softmax(dim=2)
        print(self)

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
        predictions = self.softmax(predictions)
        return predictions[-1]



def train_online(model, train_sequence, epochs=15, batch_size=1):
    num_batches = len(train_sequence)//batch_size
    print(f"Total epochs: {epochs}, total batches {num_batches}")
    pred = []
    variation =0
    correct = 0
    above = 0
    below = 0
    total = 0
    global pred_time
    global train_time
    global pred_samples
    global train_samples
    o_pred_sum = 0
    u_pred_sum = 0
    warmup = 5
    for j in range(num_batches):
        model.train()
        batch_seq = train_sequence[j*batch_size:(j+1)*batch_size]
        #split the inputs and labels
        list_inputs, list_labels = list(zip(*batch_seq))
        inputs = torch.stack(list_inputs)
        labels = torch.stack(list_labels)
        start = count()
        optimizer.zero_grad()
        #This seems to be needed in order to make the dimensions match, its just a dummy dimension
        outputs = torch.unsqueeze(model(inputs), 0)
        if torch.argmax(outputs) < torch.argmax(labels):
            train_label = torch.zeros(1,1,sizes)
            train_label[0][0][min(15,torch.argmax(labels)+1)] = 1
            single_loss = loss_function(outputs, labels)
            single_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            outputs = torch.unsqueeze(model(inputs), 0)
            single_loss = loss_function(outputs, labels)
            single_loss.backward()
            optimizer.step()
        else:
            single_loss = loss_function(outputs, labels)
            single_loss.backward()
            optimizer.step()
        train_time += count_end() - start
        train_samples += 1
        #Predict for the next timestep
        model.eval()
        with torch.no_grad():
            if j !=num_batches-1:
                batch_seq = train_sequence[(j+1)*batch_size:(j+2)*batch_size]
                list_inputs, list_labels = list(zip(*batch_seq))
                inputs = torch.stack(list_inputs)
                labels = torch.stack(list_labels)
                label_score = torch.argmax(labels)
                start = count()
                tag_score = torch.argmax(model(inputs))
                pred_time += count_end() - start
                pred_samples += 1
                buf_score = int(math.ceil(tag_score*buf))
                if buf_score > 16:
                    buf_score = 16
                print(f"Pred: {tag_score}, buffer: {buf_score}, label: {label_score}")
                if label_score < tag_score:
                    if total >= warmup: 
                        above += 1
                        o_pred_sum += tag_score - label_score
                elif label_score > buf_score:
                    if total >= warmup: 
                        below += 1
                        u_pred_sum += label_score - buf_score
                else:
                    if total >= warmup: 
                        correct += 1
                total += 1
                variation += abs(tag_score - labels)

    total-=warmup
    print(f"overpred: {above/total}, correct: {correct/total}, underpred: {below/total}")
    print(f"avg_overpred: {o_pred_sum} {above}")
    print(f"avg_underpred: {u_pred_sum} {below}")
    #print(f"time per: {(end-start)/(epochs*num_batches)}")
    return pred

def preprocessing(input_data, win):
    inout_seq = []
    L = len(input_data)
    for i in range(L-win):
        #sends in the input bandwidths for the window length
        train_seq = torch.FloatTensor(input_data[i:i+win]).to(DEVICE)
        train_label= torch.FloatTensor(input_data[i+win:i+win+1]).to(DEVICE)
        #See the input format below, pulling out the sum_tot_bw
        temp_label = torch.split(train_label, 1, dim=1)[3]
        train_label = torch.zeros(1,sizes)
        train_label[0][min(int(temp_label[0][0]),sizes-1)] = 1.0
        inout_seq.append((train_seq, train_label))
    return inout_seq


def read_input(in_file):
    #Input format:
    #max_r_delta, min_r_delta, avg_r_delta, med_r_delta, std_r_delta, allocation bucket
    input_data = []
    #Open the input data file and read everything in
    with open(in_file, "r") as f:
        for line in f:
            temp = line.split(",")
            temp = list(map(float,temp))
            input_data.append([temp[0], temp[1], temp[2], temp[5]])
    return input_data


window = 3
epochs = 15
batch_size=1 
MB = 1000000
channel_bw = 64
sizes = 16
cyc_overhead = 350

pred_time = 0
pred_samples = 0
train_time = 0
train_samples = 0
buf = 1
buffers = [1, 1.05, 1.1, 1.2, 1.3, 1.4]
if len(sys.argv) >= 3:
    buf = buffers[int(sys.argv[2])]

model = LSTM(input_size=4, output_size=sizes).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.040)
loss_function = nn.MSELoss()
#Pull the sequence of total/r/w bws from the stats file
input_data = read_input(in_file)

input_seq = preprocessing(input_data, window)

#init lstm

num_samples = len(input_seq)
print(f"num_samples: {num_samples}")

pred_online = train_online(model, input_seq, epochs=epochs, batch_size=batch_size)

#Adjust for cycle overhead of the tracking itself
train_over_adjusted = train_time - (train_samples * cyc_overhead)
train_over_adjusted/=train_samples
pred_over_adjusted = pred_time - (pred_samples * cyc_overhead)
pred_over_adjusted/=pred_samples
print(f"Training Overhead: Samples: {train_samples} Time: {train_time} Per: {train_over_adjusted}")
print(f"Prediction Overhead: Samples: {pred_samples} Time: {pred_time} Per: {pred_over_adjusted}")

checkpoint = {'model':model, 
        'state_dict': model.state_dict(),
        'optimizer':optimizer.state_dict()}
torch.save(checkpoint, 'gl_sz_size.model')
