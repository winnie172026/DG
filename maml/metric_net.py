import torch
import torch.nn as nn
class metric_net(nn.Module):

    def __init__(self, channel_in, channel_1, channel_2):
        super(metric_net, self).__init__()
        self.fc1 = nn.Linear(channel_in, channel_1)
        self.fc2 = nn.Linear(channel_1, channel_2)
        self.leaky_relu = nn.LeakyReLU()
    def forward(self, x):
        out = self.fc1(x)
        out = self.leaky_relu(out)
        out = self.fc2(out)
        out = self.leaky_relu(out)
        return out

def get_metric_net(channel_in=96, channel_1=48, channel_2=1):
    return metric_net(channel_in, channel_1, channel_2)