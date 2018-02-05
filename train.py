import os
import pdb
import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models as model_zoo
from PIL import Image
from datetime import datetime
from models import fcn8vgg
from data_loader import ADE20KLoader

USE_CUDA = torch.cuda.is_available()
num_classes = 3193
vgg16 = model_zoo.vgg16(pretrained=False)
weight_path = "./vgg16-397923af.pth"
vgg16.load_state_dict(torch.load(weight_path))

net = fcn8vgg(vgg16,num_classes)

def run(n_epoch=100, batch_size=1, print_every=1):

    data_path = './data/ADE20K_2016_07_26/'
    test_data = ADE20KLoader(data_path, num_classes,
                             types="validation",
                             is_transform=True,
                             img_size=224)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    train_data = ADE20KLoader(data_path, num_classes,
                              types="training",
                              is_transform=True,
                              img_size=224)
    train_loader = DataLoader(train_data, batch_size=batch_size)

    small_data = ADE20KLoader(data_path, num_classes,           # my valid data
                              types="small",
                              is_transform=True,
                              img_size=224)
    small_loader = DataLoader(small_data, batch_size=batch_size)


    criterion = nn.NLLLoss2d()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)
    curr_counts = 0
    for e in range(n_epoch):
        train_loss = 0
        for data in train_loader:
            im = Variable(data[0])
            im = im.cuda() if USE_CUDA else im
            label = Variable(data[1])
            label = label.cuda() if USE_CUDA else label

            # forward
            out = net(im)
            out = F.log_softmax(out, dim=1)
            # pdb.set_trace()
            loss = criterion(out, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            curr_counts += batch_size

            if curr_counts%print_every == 0: # print loss every const imgs
                eval_loss = 0
                eval_counts = 0
                for data in small_loader:
                    im = Variable(data[0], volatile=True)
                    label = Variable(data[1], volatile=True)
                    im = im.cuda() if USE_CUDA else im
                    label = label.cuda() if USE_CUDA else label

                    # forward
                    out = net(im)
                    out = F.log_softmax(out, dim=1)
                    loss = criterion(out, label)
                    eval_loss += loss.data[0]
                    eval_counts += batch_size
                print("train_loss:{0}, valid_loss:{1}".format(
                      train_loss*1.0/curr_counts, eval_loss/eval_counts))







if __name__ == "__main__":
    run()