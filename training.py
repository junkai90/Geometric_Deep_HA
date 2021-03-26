import torch
from gncc import GNCC
from torch_geometric.data import DataLoader

def train(data, lr_rate=0.01, epoch=200, batch=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNCC().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=5e-4)

    train_data = DataLoader(data, batch, True)

    model.train()

    for e in range(1,epoch+1):
        epoch_loss = 0
        for dat in train_data:
            dat = dat.to(device)
            optimizer.zero_grad()
            out = model(dat)
            loss = criterion(dat.x, dat.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    print('epoch {}, training loss {}'.format(e, epoch_loss))
    return model


