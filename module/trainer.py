import torch as tc
import torch.nn
import torch.utils.data as Data
from torch.autograd import Variable

from module.Seq2Seq import *
from module.GCN import *
from module.GAt import *
from utils.device_data_loader import *
from evaluations.metrics import *


def bulid_model(para1, para2, para3):
    
    gcn = GCN(para1[0], para1[1], para1[2])
    encoder = Encoder(para2)
    decoder = Decoder(para3)

    return [gcn, encoder, decoder]
    
def train_model(train, val, models, M, hyper_para, time_slag):
    epoches, batch_size, opti, early_stop_times, device = hyper_para
    A, L_ch = M
    n_obs, n_pred = time_slag

    min_mae = float("inf")

    models = [to_device(model, device) for model in models]

    train_data = Data.TensorDataset(
        train[:, 0:n_obs, :, :], train[:, n_obs:, :, :])

    train_loader = Data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True)

    train_loader = DeviceDataloader(train_loader, device)

    loss_func = nn.MSELoss()

    if opti == "RMSprop":
        opt_gcn = tc.optim.RMSprop(models[0].parameters())
        opt_enc = tc.optim.RMSprop(models[1].parameters())
        opt_dec = tc.optim.RMSprop(models[2].parameters())
    if opti == "Adam":
        opt_gcn = tc.optim.Adam(models[0].parameters())
        opt_enc = tc.optim.Adam(models[1].parameters())
        opt_dec = tc.optim.Adam(models[2].parameters())

    for epoch in range(epoches):
        for step, (x, y) in enumerate(train_loader):
            #print(x.shape, y.shape)
            ################################################################

            out_gcn = models[0](x, A, L_ch)

            b, t, n, c = out_gcn.shape
            out_gcn = out_gcn.permute(0, 2, 1, 3).contiguous().view(-1, t, c)
            ################################################################
            out_enc, hidden = models[1](out_gcn)
            ################################################################
            b_y, t_y, n_y, c_y = y.shape
            y = y.permute(0, 2, 1, 3).contiguous().view(-1, t_y, c_y)
            target = tc.zeros_like(y[:, 0:1, :])
            for i in range(n_pred):

                out_dec, hidden_dec = models[2](target, hidden)
                if i == 0:
                    outs_dec = out_dec
                else:
                    outs_dec = tc.cat([outs_dec, out_dec], dim=1)
                target = out_dec
                hidden = hidden_dec

            ################################################################
            
            loss = loss_func(outs_dec, y)

            opt_gcn.zero_grad()
            opt_enc.zero_grad()
            opt_dec.zero_grad()

            loss.backward()

            opt_dec.step()
            opt_enc.step()
            opt_gcn.step()
            print(f"epoch: {epoch}, setp: {step}, loss: {loss.data}")

        mae = MAE(val, models, [A, L_ch], [
                  n_obs, n_pred], [batch_size, device])
        mape = MAPE(val, models, [A, L_ch], [
            n_obs, n_pred], [batch_size, device])
        rmse = RMSE(val, models, [A, L_ch], [
            n_obs, n_pred], [batch_size, device])
        ###################################################################
        cur_mae = mae.item()  # early_stop
        if cur_mae < min_mae:
            min_mae = cur_mae
            count = 0
        else:
            count += 1
            if count > early_stop_times:
                print("early_stop")
                break
        #####################################################################
        print(
            f"epoch: {epoch}, MAE: {mae.data}, MAPE: {mape.data}, RMSE: {rmse.data}")
    return models
