import torch
import torch.nn.functional as F
from torch_geometric.nn import SplineConv, BatchNorm

class GNCC(torch.nn.Module):
    def __init__(self):
        super(GNCC, self).__init__()
        self.conv1 = SplineConv(1,8,3, kernel_size=25)
        self.batch1 = BatchNorm(8)
        self.conv2 = SplineConv(8,16,3, kernel_size=25)
        self.batch2 = BatchNorm(16)
        self.conv3 = SplineConv(16,32,3, kernel_size=25)
        self.batch3 = BatchNorm(32)
        self.conv4 = SplineConv(32,32,3, kernel_size=25)
        self.batch4 = BatchNorm(32)
        self.conv5 = SplineConv(32,32,3, kernel_size=25)
        self.batch5 = BatchNorm(32)
        self.conv6 = SplineConv(32,32,3, kernel_size=25)
        self.batch6 = BatchNorm(32)
        self.conv7 = SplineConv(32,32,3, kernel_size=25)
        self.batch7 = BatchNorm(32)
        self.conv8 = SplineConv(32,32,3, kernel_size=25)
        self.batch8 = BatchNorm(32)
        self.conv9 = SplineConv(32,32,3, kernel_size=25)
        self.batch9 = BatchNorm(32)
        self.conv10 = SplineConv(32,16,3, kernel_size=25)
        self.batch10 = BatchNorm(16)
        self.conv11 = SplineConv(16,8,3, kernel_size=25)
        self.batch11 = BatchNorm(8)
        self.conv12 = SplineConv(8,1,3, kernel_size=25)

    def forward(self, data):
        x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x,edge_index,pseudo)
        x = F.elu(x)
        x = self.batch1(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x,edge_index,pseudo)
        x = F.elu(x)
        x = self.batch2(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv3(x,edge_index,pseudo)
        x = F.elu(x)
        x = self.batch3(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv4(x,edge_index,pseudo)
        x = F.elu(x)
        x = self.batch4(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv5(x,edge_index,pseudo)
        x = F.elu(x)
        x = self.batch5(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv6(x,edge_index,pseudo)
        x = F.elu(x)
        x = self.batch6(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv7(x,edge_index,pseudo)
        x = F.elu(x)
        x = self.batch7(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv8(x,edge_index,pseudo)
        x = F.elu(x)
        x = self.batch8(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv9(x,edge_index,pseudo)
        x = F.elu(x)
        x = self.batch9(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv10(x,edge_index,pseudo)
        x = F.elu(x)
        x = self.batch10(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv11(x,edge_index,pseudo)
        x = F.elu(x)
        x = self.batch11(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv1(x,edge_index,pseudo)

        return x



