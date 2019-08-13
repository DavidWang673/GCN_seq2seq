import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import torch as tc


from utils.data_utils import *
from utils.device_data_loader import *
from module.trainer import*

n_obs, n_pred = 12, 6

data_path = "../data/V_228.csv"
train, val, test = data_gen(data_path, n_obs, n_pred)

train = tc.from_numpy(train).float()
val = tc.from_numpy(val).float()
test = tc.from_numpy(test).float()

node_path = "../data/W_228.csv"
A = pd.read_csv(node_path, header=None).values
A = scaled_adjacency(A)
A = tc.from_numpy(A).float().cuda()
# print(train.shape, A.shape)

channel_in, Ks, gcn_out_size = train.shape[-1], 3, 8

L_ch = tc.from_numpy(chebyshev_polynomials(A, Ks)).float().cuda()
A = tc.from_numpy(A).float().cuda()

encoder_input_size, hidden_size, decoder_input_size = gcn_out_size, 16, channel_in


device = get_default_device()

models = bulid_model([channel_in, Ks, gcn_out_size], [
                     encoder_input_size, hidden_size], [decoder_input_size, hidden_size])


epoches, batch_size, opti, early_stop_times = 300, 50, "Adam", 20  # "RMSprop","Adam"
hyper_para = [epoches, batch_size, opti, early_stop_times, device]

models = train_model(train, val, models, [
                     A, L_ch], hyper_para, [n_obs, n_pred])
tc.save(models[0], "../models/GCN.pkl")
tc.save(models[1], "../models/encoder.pkl")
tc.save(models[2], "../models/decoder.pkl")
