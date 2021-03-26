import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
from pytorchtools import EarlyStopping

class dataset(Dataset):
    def __init__(self, X, Y):
        self.Y = Y
        self.X = X

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

def train_nn(X, Y, net, lr_rate=0.01, Lambda=0.1, epoch=500, batch=64):
    if torch.cuda.is_available():
        net = net.cuda()
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr_rate) 

    #x_data = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    x_data = torch.from_numpy(X).float()
    y_data = torch.from_numpy(Y).float()

    train_dataset = dataset(x_data, y_data)
    train_data = DataLoader(train_dataset, batch, True)

    print('Start training')
    for epoch in range(1, epoch+1):

        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_L1_loss = 0
        for x, y in train_data: 
            if torch.cuda.is_available():
                x = Variable(x.cuda())
                y = Variable(y.cuda())
            else:
                x = Variable(x)
                y = Variable(y)
    
            # Forward pass: Compute predicted y by passing  
            # x to the model 
            pred_y = net(x) 

            L1_loss = 0
            for param in net.parameters():
                L1_loss += torch.sum(torch.abs(param))
            # Compute and print loss
            mse_loss = criterion(pred_y, y)
            loss = mse_loss + Lambda * L1_loss

            # Zero gradients, perform a backward pass,  
            # and update the weights. 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_L1_loss += L1_loss.item()

        print('epoch {}, loss {}, mse_loss {}, L1_loss {}'.format(epoch, epoch_loss, epoch_mse_loss, Lambda*epoch_L1_loss))
    return net

def train_nn_with_test_loss(X, Y, net, lr_rate=0.01, Lambda=0.1, epoch=500, batch=64, test_dict={}):
    test_loss = []
    
    early_stopping = EarlyStopping(patience=300, verbose=True)

    if torch.cuda.is_available():
        net = net.cuda()
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr_rate) 

    #x_data = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    x_data = torch.from_numpy(X).float()
    y_data = torch.from_numpy(Y).float()

    train_dataset = dataset(x_data, y_data)
    train_data = DataLoader(train_dataset, batch, True)

    x_test = torch.from_numpy(test_dict['x_test']).float()
    y_true = test_dict['y_true']
    y_mean = test_dict['y_mean']
    y_norm = test_dict['y_norm']

    if torch.cuda.is_available():
        x_test = Variable(x_test.cuda())
    else:
        x_test = Variable(x_test)

    print('Start training')
    for epoch in range(1, epoch+1):
        net.train()
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_L1_loss = 0
        for x, y in train_data: 
            if torch.cuda.is_available():
                x = Variable(x.cuda())
                y = Variable(y.cuda())
            else:
                x = Variable(x)
                y = Variable(y)
    
            # Forward pass: Compute predicted y by passing  
            # x to the model 
            pred_y = net(x) 

            L1_loss = 0
            for param in net.parameters():
                L1_loss += torch.sum(torch.abs(param))
            # Compute and print loss
            mse_loss = criterion(pred_y, y)
            loss = mse_loss + Lambda * L1_loss

            # Zero gradients, perform a backward pass,  
            # and update the weights. 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_L1_loss += L1_loss.item()

        print('-'*70)
        print('epoch {}, train loss: {} (mse_loss {}, L1_loss {})'.format(epoch, epoch_loss, epoch_mse_loss, Lambda*epoch_L1_loss))

        net.eval()
        y_pred = net(x_test)
        y_pred = y_pred.detach().cpu().numpy()

        y_pred = y_pred * y_norm + y_mean
        test_mse = mean_squared_error(y_true, y_pred)
        print('Test loss: {}'.format(test_mse))

        n_units = y_true.shape[1]
        accuracy = np.array([np.corrcoef(y_pred[:, i].flatten(), y_true[:, i].flatten())[0, 1]
                             for i in range(n_units)])
        accuracy = accuracy.reshape((1,) + y_pred.shape[1:])
        mean_acc = np.mean(accuracy)
        print('Mean prediction accuracy: {}'.format(mean_acc))


        early_stopping(-mean_acc, net)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

    net.load_state_dict(torch.load('checkpoint.pt'))
    return net




def train_nn_with_test_loss_wTw(X, Y, net, lr_rate=0.01, Lambda=0.1, epoch=500, batch=64, test_dict={}):
    test_loss = []
    
    early_stopping = EarlyStopping(patience=300, verbose=True)

    if torch.cuda.is_available():
        net = net.cuda()
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr_rate) 

    #x_data = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    x_data = torch.from_numpy(X).float()
    y_data = torch.from_numpy(Y).float()

    train_dataset = dataset(x_data, y_data)
    train_data = DataLoader(train_dataset, batch, True)

    x_test = torch.from_numpy(test_dict['x_test']).float()
    y_true = test_dict['y_true']
    y_mean = test_dict['y_mean']
    y_norm = test_dict['y_norm']

    if torch.cuda.is_available():
        x_test = Variable(x_test.cuda())
    else:
        x_test = Variable(x_test)

    print('Start training')
    for epoch in range(1, epoch+1):
        net.train()
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_L1_loss = 0
        for x, y in train_data: 
            if torch.cuda.is_available():
                x = Variable(x.cuda())
                y = Variable(y.cuda())
            else:
                x = Variable(x)
                y = Variable(y)
    
            # Forward pass: Compute predicted y by passing  
            # x to the model 
            pred_y = net(x) 

            L1_loss = 0
            for param in net.parameters():
                L1_loss += torch.sum(torch.abs(param))
            # Compute and print loss
            mse_loss = criterion(pred_y, y)
            loss = mse_loss + Lambda * L1_loss

            # Zero gradients, perform a backward pass,  
            # and update the weights. 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_L1_loss += L1_loss.item()

        print('-'*70)
        print('epoch {}, train loss: {} (mse_loss {}, L1_loss {})'.format(epoch, epoch_loss, epoch_mse_loss, Lambda*epoch_L1_loss))

        net.eval()
        y_pred = net(x_test)
        y_pred = y_pred.detach().cpu().numpy()

        y_pred = y_pred * y_norm + y_mean
        test_mse = mean_squared_error(y_true, y_pred)
        print('Test loss: {}'.format(test_mse))

        n_units = y_true.shape[1]
        accuracy = np.array([np.corrcoef(y_pred[:, i].flatten(), y_true[:, i].flatten())[0, 1]
                             for i in range(n_units)])
        accuracy = accuracy.reshape((1,) + y_pred.shape[1:])
        mean_acc = np.mean(accuracy)
        print('Mean prediction accuracy: {}'.format(mean_acc))


        early_stopping(-mean_acc, net)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

    net.load_state_dict(torch.load('checkpoint.pt'))
    return net