import torch as tc
import torch.nn
import torch.utils.data as Data
from utils.device_data_loader import *


def MAE(val_data, models, M, time_slag, hyper_para):
    b, t, n, c = val_data.shape
    A, L_ch = M
    n_obs, n_pred = time_slag
    batch_size, device = hyper_para

    models = [model.eval() for model in models]

    torch_val_data = Data.TensorDataset(
        val_data[:, 0:n_obs, :, :], val_data[:, n_obs:, :, :])
    val_loader = Data.DataLoader(
        dataset=torch_val_data, batch_size=batch_size, shuffle=True)
    val_loader = DeviceDataloader(val_loader, device)

    for step, (x, y) in enumerate(val_loader):
        ################################################################
        with tc.no_grad():
            out_gcn = models[0](x, A, L_ch)

        b, t, n, c = out_gcn.shape
        out_gcn = out_gcn.permute(0, 2, 1, 3).contiguous().view(-1, t, c)
        ################################################################
        with tc.no_grad():
            out_enc, hidden = models[1](out_gcn)
        ################################################################
        b_y, t_y, n_y, c_y = y.shape
        y = y.permute(0, 2, 1, 3).contiguous().view(-1, t_y, c_y)
        target = tc.zeros_like(y[:, 0:1, :])
        for i in range(n_pred):
            with tc.no_grad():
                out_dec, hidden_dec = models[2](target, hidden)

            if step == 0 and i == 0:
                loss = tc.abs(out_dec - y[:, i:i + 1, :])
            else:
                if loss.shape[0] == out_dec.shape[0]:
                    loss += tc.abs(out_dec - y[:, i:i + 1, :])
                else:
                    loss = tc.cat(
                        [loss, tc.abs(out_dec - y[:, i:i + 1, :])], dim=0)
            target = out_dec
            hidden = hidden_dec

        ################################################################
        # print(KKKKKKKK)
        # print(outs_dec.shape, y.shape)
    loss = tc.sum(loss) / (b * t * n * c)
    models = [model.train() for model in models]
    return loss


def RMSE(val_data, models, M, time_slag, hyper_para):
    b, t, n, c = val_data.shape
    A, L_ch = M
    n_obs, n_pred = time_slag
    batch_size, device = hyper_para

    models = [model.eval() for model in models]

    torch_val_data = Data.TensorDataset(
        val_data[:, 0:n_obs, :, :], val_data[:, n_obs:, :, :])
    val_loader = Data.DataLoader(
        dataset=torch_val_data, batch_size=batch_size, shuffle=True)
    val_loader = DeviceDataloader(val_loader, device)

    for step, (x, y) in enumerate(val_loader):
        ################################################################
        with tc.no_grad():
            out_gcn = models[0](x, A, L_ch)

        b, t, n, c = out_gcn.shape
        out_gcn = out_gcn.permute(0, 2, 1, 3).contiguous().view(-1, t, c)
        ################################################################
        with tc.no_grad():
            out_enc, hidden = models[1](out_gcn)
        ################################################################
        b_y, t_y, n_y, c_y = y.shape
        y = y.permute(0, 2, 1, 3).contiguous().view(-1, t_y, c_y)
        target = tc.zeros_like(y[:, 0:1, :])
        for i in range(n_pred):
            with tc.no_grad():
                out_dec, hidden_dec = models[2](target, hidden)

            if step == 0 and i == 0:
                loss = (out_dec - y[:, i:i + 1, :])**2
            else:
                if loss.shape[0] == out_dec.shape[0]:
                    loss += (out_dec - y[:, i:i + 1, :])**2
                else:
                    loss = tc.cat(
                        [loss, (out_dec - y[:, i:i + 1, :])**2], dim=0)
            target = out_dec
            hidden = hidden_dec

        ################################################################
    loss = (tc.sum(loss) / (b * t * n * c))**0.5
    models = [model.train() for model in models]
    return loss


def MAPE(val_data, models, M, time_slag, hyper_para):
    b, t, n, c = val_data.shape
    A, L_ch = M
    n_obs, n_pred = time_slag
    batch_size, device = hyper_para

    models = [model.eval() for model in models]

    torch_val_data = Data.TensorDataset(
        val_data[:, 0:n_obs, :, :], val_data[:, n_obs:, :, :])
    val_loader = Data.DataLoader(
        dataset=torch_val_data, batch_size=batch_size, shuffle=True)
    val_loader = DeviceDataloader(val_loader, device)

    for step, (x, y) in enumerate(val_loader):
        ################################################################
        with tc.no_grad():
            out_gcn = models[0](x, A, L_ch)

        b, t, n, c = out_gcn.shape
        out_gcn = out_gcn.permute(0, 2, 1, 3).contiguous().view(-1, t, c)
        ################################################################
        with tc.no_grad():
            out_enc, hidden = models[1](out_gcn)
        ################################################################
        b_y, t_y, n_y, c_y = y.shape
        y = y.permute(0, 2, 1, 3).contiguous().view(-1, t_y, c_y)
        target = tc.zeros_like(y[:, 0:1, :])
        for i in range(n_pred):
            with tc.no_grad():
                out_dec, hidden_dec = models[2](target, hidden)

            if step == 0 and i == 0:
                loss = tc.abs((out_dec - y[:, i:i + 1, :]) / y[:, i:i + 1, :])
            else:
                if loss.shape[0] == out_dec.shape[0]:
                    loss += tc.abs((out_dec -
                                    y[:, i:i + 1, :]) / y[:, i:i + 1, :])
                else:
                    loss = tc.cat(
                        [loss, tc.abs((out_dec - y[:, i:i + 1, :]) / y[:, i:i + 1, :])], dim=0)
            target = out_dec
            hidden = hidden_dec

        ################################################################
    loss = (tc.sum(loss) / (b * t * n * c)) * 100
    models = [model.train() for model in models]
    return loss
