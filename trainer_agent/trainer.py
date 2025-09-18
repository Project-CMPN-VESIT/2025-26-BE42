import torch, torch.optim as optim
import torch.nn as nn
from trainer_agent.model import DummyModel

def compute_weight_delta(before, after):
    delta = {}
    for k in before.keys():
        delta[k] = (after[k].cpu() - before[k].cpu())
    return delta

def train_model(X, y, input_dim, epochs=1, batch_size=32, lr=1e-3, device="cpu"):
    model = DummyModel(input_dim=input_dim).to(device)

    before = {k: v.clone().detach().cpu() for k,v in model.state_dict().items()}

    dataset = torch.utils.data.TensorDataset(X.to(device), y.to(device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            # dummy loss (MSE towards zero vector)
            loss = (out**2).mean()
            loss.backward()
            opt.step()

    after = {k: v.clone().detach().cpu() for k,v in model.state_dict().items()}
    delta = compute_weight_delta(before, after)
    return delta, model
