"""models.py - neuronske mreze (PyTorch): MLP i LSTM, te zajednicka petlja treniranja."""
import copy
import time
import numpy as np
import torch
from torch import nn

torch.set_num_threads(2)


class MLP(nn.Module):
    """Viseslojni perceptron: F -> h1 -> h2 -> ... -> 1, ReLU i dropout u skrivenim slojevima."""

    def __init__(self, n_in, hidden=(128, 64, 32), dropout=0.1):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LSTMReg(nn.Module):
    """LSTM regresor: prozor (N, w, F) -> skriveno stanje h_t posljednjeg koraka -> P_hat."""

    def __init__(self, n_in, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model, X, batch=4096):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            outs.append(model(torch.as_tensor(X[i:i + batch], dtype=torch.float32)).numpy())
    return np.concatenate(outs)


def train_nn(model, Xtr, ytr, Xva, yva, lr=1e-3, weight_decay=1e-5, batch=256,
             max_epochs=200, patience=15, seed=0, verbose=False, lr_schedule=None):
    """Adam + MSE gubitak, rano zaustavljanje po validacionom MSE, smanjenje stope ucenja.
    Ako je zadat lr_schedule (lista stopa ucenja po epohama), model se trenira tacno
    len(lr_schedule) epoha bez validacije (ponovno treniranje na treningu + validaciji)."""
    set_seed(seed)
    if lr_schedule is not None:
        return _train_fixed(model, Xtr, ytr, lr_schedule, weight_decay, batch, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    loss_fn = nn.MSELoss()
    Xt = torch.as_tensor(Xtr, dtype=torch.float32)
    yt = torch.as_tensor(ytr, dtype=torch.float32)
    best, best_state, wait = np.inf, None, 0
    hist = {"train": [], "val": []}
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        tr_mse = tot / len(Xt)
        va_mse = float(np.mean((predict(model, Xva) - yva) ** 2))
        hist["train"].append(tr_mse)
        hist.setdefault("lr", []).append(opt.param_groups[0]["lr"])
        hist["val"].append(va_mse)
        sched.step(va_mse)
        if va_mse < best - 1e-7:
            best, best_state, wait = va_mse, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose:
            print(f"  ep {ep+1:3d} train {tr_mse:.5f} val {va_mse:.5f}")
    model.load_state_dict(best_state)
    hist["best_epoch"] = int(np.argmin(hist["val"]) + 1)
    hist["epochs"] = len(hist["val"])
    hist["seconds"] = time.time() - t0
    return model, hist


def _train_fixed(model, Xtr, ytr, lr_schedule, weight_decay, batch, seed):
    opt = torch.optim.Adam(model.parameters(), lr=lr_schedule[0], weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    Xt = torch.as_tensor(Xtr, dtype=torch.float32)
    yt = torch.as_tensor(ytr, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    hist = {"train": [], "lr": list(lr_schedule)}
    t0 = time.time()
    for lr in lr_schedule:
        for pg in opt.param_groups:
            pg["lr"] = lr
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        hist["train"].append(tot / len(Xt))
    hist["epochs"] = len(lr_schedule)
    hist["seconds"] = time.time() - t0
    return model, hist
