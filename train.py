import torch
import torchvision
from tqdm import tqdm
import os
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description='train q network')
parser.add_argument('-g',
                    '--gpu',
                    dest='gpu',
                    default='0',
                    help='which gpu to run on')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

conv_size = 28 - 4 - 4
latent_size = 16
hidden_size = 512
flatten_size = conv_size * conv_size * 64
normal_distrib = torch.distributions.normal.Normal(0, 1)
keep_prob = 0.9


class EncoderFC(nn.Module):
    def __init__(self):
        super(EncoderFC, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, latent_size * 2)

    def forward(self, inp):
        x = inp.view(-1, 28 * 28)
        x = F.dropout(F.relu(self.fc1(x)),keep_prob)
        x = F.dropout(F.relu(self.fc2(x)),keep_prob)
        x = self.fc3(x)
        epsilon = normal_distrib.sample((8, latent_size))
        means = x[:, :latent_size]
        stds = x[:, latent_size:]
        out = means + stds * epsilon.cuda()

        # derived from
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        kld = 0.5 * (stds**2 + means**2 - 1 - (stds**2 + 1e-8).log())
        return out, kld


class DecoderFC(nn.Module):
    def __init__(self):
        super(DecoderFC, self).__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 28 * 28)

    def forward(self, inp):
        x = F.dropout(F.relu(self.fc1(inp)),keep_prob)
        x = F.dropout(F.relu(self.fc2(x)),keep_prob)
        x = torch.sigmoid(self.fc3(x))
        x = x.view((-1, 1, 28, 28))
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.conv1 = nn.Conv2d(1, 16, 5)
        # self.conv2 = nn.Conv2d(16, 64, 5)
        # self.fc1 = nn.Linear(flatten_size, 64)
        # self.fc2 = nn.Linear(64, 32)

        self.conv1 = nn.Conv2d(1, 64, 4, stride=2)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 4, stride=2)
        self.fc = nn.Linear(64, latent_size * 2)

    def forward(self, inp):
        x = F.dropout(F.relu(self.conv1(inp)))
        x = F.dropout(F.relu(self.conv2(x)))
        x = F.dropout(F.relu(self.conv3(x)))
        x = x.view(-1, 64)
        x = self.fc(x)
        epsilon = normal_distrib.sample((8, latent_size))
        means = x[:, :latent_size]
        stds = x[:, latent_size:]
        out = means + stds * epsilon.cuda()

        # derived from
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        kld = 0.5 * (stds**2 + means**2 - 1 - (stds**2 + 1e-8).log())
        return out, kld


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_size, 64)
        self.dconv1 = nn.ConvTranspose2d(64, 64, 5, stride=2)
        self.dconv2 = nn.ConvTranspose2d(64, 64, 5, stride=2)
        self.dconv3 = nn.ConvTranspose2d(64, 1, 4, stride=2)
        self.fc2 = nn.Linear(28 * 28, 28 * 28)

    def forward(self, inp):
        x = F.dropout(F.relu(self.fc(inp)))
        x = x.view(-1, 64, 1, 1)
        x = F.dropout(F.relu(self.dconv1(x)))
        x = F.dropout(F.relu(self.dconv2(x)))
        x = F.dropout(F.relu(self.dconv3(x)))
        x = x.view((-1, 28**2))
        x = torch.sigmoid(self.fc2(x))
        x = x.view((-1, 1, 28, 28))
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = EncoderFC()
        self.decoder = DecoderFC()

    def forward(self, x):
        x, kld = self.encoder(x)
        x = self.decoder(x)
        return x, kld


batch_size = 8
train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
    './data',
    download=True,
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])),
                                           batch_size=batch_size,
                                           shuffle=True)

net = Net()
net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(10):
    running_loss = 0.0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()
        outputs, kld = net(inputs)
        im_loss = F.binary_cross_entropy(outputs, inputs, reduction='sum') / batch_size
        loss = im_loss + kld.sum() / batch_size
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

import pdb
pdb.set_trace()
inp = normal_distrib.sample((8, latent_size)).cuda()
outs = net.decoder(inp)
ins, labs = next(iter(test_loader))
ot, kld = net(ins.cuda())
torchvision.utils.save_image(ins, './ins.png')
torchvision.utils.save_image(ot, './ot.png')
torchvision.utils.save_image(outs, './outs.png')

print('Finished Training')
