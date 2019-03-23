import torch
import torch.utils.data as Data
import torchvision
from lib.network import Network
from torch import nn
import time
import argparse
import numpy as np
import os


def run(args):

    x_dim = [76, 91, 75]
    y_dim = 10
    splits = [0.8, 0.1, 0.1]

    X_data = []
    Y_data = []

    num_examples = 10

    for i in range(1, num_examples+1):
        f_x = os.path.join(args.dataset, f'X_Img_Values{i}.npy')
        f_y = os.path.join(args.dataset, f'YValues{i}.npy')

        X_data.append(torch.tensor(np.load(f_x)))
        Y_data.append(torch.tensor(np.load(f_y)))


    # (batch, channel, width, height)
    X_data = torch.cat(X_data, dim=0).view([-1, x_dim[2], *x_dim[:2]])
    Y_data = torch.cat(Y_data, dim=0).view([-1, y_dim])

    assert X_data.shape[0] == Y_data.shape[0]
    total_size = X_data.shape[0]

    X_train = X_data[:int(total_size*splits[0]), ...]
    Y_train = Y_data[:int(total_size*splits[0]), ...]

    X_dev = X_data[int(total_size*splits[0]):int(total_size*(splits[0]+splits[1])), ...]
    Y_dev = Y_data[int(total_size*splits[0]):int(total_size*(splits[0]+splits[1])), ...]

    X_test = X_data[int(total_size*(splits[0]+splits[1])):, ...]
    Y_test = Y_data[int(total_size*(splits[0]+splits[1])):, ...]


    train_dataset = Data.TensorDataset(X_train, Y_train)
    dev_dataset = Data.TensorDataset(X_dev, Y_dev)
    test_dataset = Data.TensorDataset(X_test, Y_test)


    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    dev_loader = Data.DataLoader(dataset=dev_dataset, batch_size=16, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)


    net = Network(in_channels=x_dim[2], out_size=y_dim)
    if torch.cuda.is_available():
        net = nn.DataParallel(net)
        net.cuda()

    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = nn.MSELoss(reduction='mean')

    for epoch_index in range(20):
        st = time.time()

        torch.set_grad_enabled(True)
        net.train()
        for train_batch_index, (img_batch, label_batch) in enumerate(train_loader):
            if torch.cuda.is_available():
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()

            predict = net(img_batch)
            loss = loss_func(predict.float(), label_batch.float())

            net.zero_grad()
            loss.backward()
            opt.step()

        print('(LR:%f) Time of a epoch:%.4fs' % (opt.param_groups[0]['lr'], time.time()-st))


        # evaluation
        torch.set_grad_enabled(False)
        net.eval()
        total_loss = []
        total_sample = 0

        for dev_batch_index, (img_batch, label_batch) in enumerate(dev_loader):
            if torch.cuda.is_available():
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()

            predict = net(img_batch)
            loss = loss_func(predict.float(), label_batch.float())

            total_loss.append(loss)
            total_sample += img_batch.size(0)

        mean_loss = sum(total_loss) / len(total_loss)

        print('[Test] epoch[%d/%d] loss:%.4f\n'
              % (epoch_index, 100, mean_loss.item()))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='dataset folder')

    args = parser.parse_args()

    run(args)

