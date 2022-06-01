import torch.nn as nn
import torch
import numpy as np
import sys
import mmap
import os
import time
import math
from macros import *

DEVICE = torch.device("cpu")

in_file = "NOT USING INPUT FILE"
global out_f
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=4, output_size=1):
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
        predictions = self.softmax(predictions)
        return predictions[-1]


def get_inputs():
    fname = f'{BLOCKFLEX_DIR}/sz_inputs.txt'
    cur_version = 0
    with open(fname, 'r') as fd:
        mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ, offset=0)
        while True:
            #Reset to the start of the file
            mm.seek(0)
            ret_inps = list(map(float, mm.read().decode("utf-8").strip().split()))
            if ret_inps[0] > cur_version and len(ret_inps) > 4:
                cur_version = ret_inps[0]
                log_msg(f"Starting Iteration: {cur_version}", out_f=out_f)
                yield ret_inps[1:]
            elif ret_inps[0] < 0:
                mm.close()
                return None
            time.sleep(1)

def train_online(model, optimizer, loss_function, buf, sz_q):

    #Store the history of inputs
    pred = []
    train_sequence = []

    #Previous prediction, used for accuracy reporting
    buf_score = None

    total = 0
    warmup = 5

    for temp_seq in get_inputs():
        train_sequence.append(temp_seq)
        if len(train_sequence) <= window: continue
        batch_seq = []
        t_seq = preprocess_single(train_sequence,window)
        if t_seq is None:
            sz_q.put(-1)
            break
        batch_seq.append(t_seq)
        #split the inputs and labels
        list_inputs, list_labels = list(zip(*batch_seq))
        inputs = torch.stack(list_inputs)
        labels = torch.stack(list_labels)
        label_score = torch.argmax(labels)
        model.train()
        optimizer.zero_grad()
        outputs = torch.unsqueeze(model(inputs), 0)

        #TRAIN
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

        #Predict for the next timestep
        model.eval()
        with torch.no_grad():
            batch_seq = []
            batch_seq.append(pred_single(train_sequence, window))
            inputs = torch.stack(list_inputs)
            labels = torch.stack(list_labels)
            tag_score = torch.argmax(model(inputs))
            buf_score = int(math.ceil(tag_score*buf))
            if buf_score > 16:
                buf_score = 16
            #print("Adding to sz q")
            sz_q.put(buf_score)
        total += 1

    return pred


def pred_single(input_data, win):
    i = len(input_data)-1
    train_seq = torch.FloatTensor(input_data[i-win+1:i+1]).to(DEVICE)
    train_label= torch.FloatTensor(input_data[i:i+1]).to(DEVICE)
    return (train_seq, train_label)

def preprocess_single(input_data, win):
    i = len(input_data)-1
    #sends in the input bandwidths for the window length
    train_seq = torch.FloatTensor(input_data[i-win:i]).to(DEVICE)
    train_label= torch.FloatTensor(input_data[i:i+1]).to(DEVICE)
    #See the input format below, pulling out the sum_tot_bw
    temp_label = torch.split(train_label, 1, dim=1)[3]
    train_label = torch.zeros(1,sizes)
    train_label[0][min(int(temp_label[0][0]),sizes-1)] = 1.0
    ret_seq = (train_seq, train_label)
    return ret_seq

sizes = 16
#Already covered elsewhere
window = 3
MB = 1000000
channel_bw = 64
def main(sz_q, f=None):
    global out_f
    out_f = f
    buf = 1.05

    model = LSTM(input_size=4, output_size=sizes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.040)
    loss_function = nn.MSELoss()

    pred_online = train_online(model, optimizer, loss_function, buf,sz_q)
