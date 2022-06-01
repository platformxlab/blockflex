import torch.nn as nn
import torch
import math
import numpy as np
import sys
from macros import *


# Here we now use the sequence of last durations we have hit (5 min increments to predict how much
# longer it will endure. This will require somewhat updating the training procedure to when
# we actually have a shift.
windows = [180]
DEVICE = torch.device("cpu")
global out_f
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
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

def train_single(model, optimizer, loss_function, t_seq):
    batch_seq = []
    batch_seq.append(t_seq)
    list_inputs, list_labels = list(zip(*batch_seq))
    inputs = torch.stack(list_inputs)
    labels = torch.stack(list_labels)
    model.train()
    optimizer.zero_grad()
    out = torch.unsqueeze(model(inputs), 0)
    single_loss = loss_function(out, labels)
    single_loss.backward()
    optimizer.step()

def pred_single(model, t_seq, harvest):
    batch_seq = []
    batch_seq.append(t_seq)
    model.eval()
    with torch.no_grad():
        list_inputs, list_labels = list(zip(*batch_seq))
        inputs = torch.stack(list_inputs)
        labels = torch.stack(list_labels)
        tag_score = model(inputs)
        tag_score_int = int(max(0,tag_score*outputs))
        if harvest:
            buf_score_int = int(math.ceil(tag_score_int*buf))
        else:
            buf_score_int = int(math.ceil(tag_score_int*(1/buf)))
        if buf_score_int > outputs:
            buf_score_int = outputs
        return buf_score_int


def get_pred(input_hist, cur_dur):
    ret_seq = list()
    while cur_dur > 0:
        ret_seq.append([cur_dur/time_alloc])
        cur_dur -= rem
        if len(ret_seq) == 3: break
       
    if len(input_hist) > 0:
        temp_list = input_hist.copy()
        temp_list.reverse()
        for it in temp_list:
            ret_seq.append(it)
            if len(ret_seq) == 3: break
    if len(ret_seq) > 0:
        ret_seq.reverse()
    return (torch.FloatTensor(ret_seq),torch.FloatTensor([[0]]))

def preprocess_single(input_hist, label_hist, win):
    i = len(input_hist)-1
    train_seq = torch.FloatTensor(input_hist[i-win:i]).to(DEVICE)
    train_label= torch.FloatTensor(label_hist[i:i+1]).to(DEVICE)
    if (train_label[0] > outputs): train_label[0] = outputs
    if (train_label[0] < 0): train_label[0] = 0
    train_label[0]/=outputs
    ret_seq = (train_seq, train_label)
    return ret_seq

def process_input(amt, duration, hist, model, optimizer, input_hist, label_hist, window, loss_function):
    #Setting up the training for ensuring we have 'i' bandwidth or less
    #This each time we have amt bandwidth we need to train for all bw < amt
    #The rest will have their durations extended.
    #Analogous for sz as for bw described above
    for i in range(channels):
        #Want to train for those with lower
        if i < amt:
            #If have hist to train with
            if hist[i] > 0:
                cur_dur = hist[i]
                hist[i] = 0
                cur = min(rem, cur_dur)
                while cur <= cur_dur:
                    input_hist[i].append([cur/time_alloc])
                    label_hist[i].append([(duration - cur)/time_alloc])
                    #Every interval
                    cur += rem
                    if len(input_hist[i]) <= window: continue
                    #Train here
                    train_input = preprocess_single(input_hist[i], label_hist[i], window)
                    train_single(model[i], optimizer[i],loss_function, train_input)
        else:
            hist[i] += duration

time_alloc = 1800
window = 3
#Receive at three minute intervals
rem = 180
buf = 1.05
outputs = 25
out_lim =outputs * time_alloc
channels = 17
def main(in_q, comb_q, combiner, harvest, f=None):
    global out_f
    out_f = f
    #What is the current history, used for generating inputs for the training from the trace
    hist = [0 for _ in range(channels)]
    input_hist = [[] for _ in range(channels)]
    label_hist = [[] for _ in range(channels)]


    #Set of models for each bandwidth allocation size
    model = []
    optimizer = []
    for i in range(channels):
        model.append(LSTM(input_size=1, output_size=1).to(DEVICE))
        optimizer.append(torch.optim.Adam(model[i].parameters(), lr=0.006))
    loss_function = nn.MSELoss()


    ##DONE SETUP, NOW RUNNING

    prev_input = None
    cur_dur = 0
    while True:
        next_input = in_q.get()
        #print("READ INPUT FROM Q: ", next_input, combiner)
        cur_dur += rem
        if next_input < 0:
            break
        if prev_input is None:
            prev_input = next_input
        if prev_input != next_input or cur_dur % time_alloc is 0:
            #Process the input and train
            process_input(prev_input, cur_dur, hist, model, optimizer, input_hist, label_hist, window, loss_function)
            cur_dur = 0
        #Predict the next value
        pred_input = get_pred(input_hist[next_input], cur_dur)
        pred = -1
        temp, _ = pred_input
        if len(temp) == 3:
            pred = pred_single(model[next_input], pred_input, harvest)
        #Combine the values accordingly
        if combiner:
            other_pred = comb_q.get()
            #Combine logic
            if pred==-1 and other_pred ==-1:
                log_msg("NO PREDICTION", out_f=out_f)
            elif pred==-1:
                log_msg("PREDICTING:", other_pred, out_f=out_f)
            elif other_pred == -1:
                log_msg("PREDICTING:", pred, out_f=out_f)
            else:
                log_msg("PREDICTING:", min(pred, other_pred), out_f=out_f)
        else:
            comb_q.put(pred)
        prev_input = next_input
