

import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
from config import get_config
import torchvision
from torch.utils.data import DataLoader

from VisionTransformer import VisionTransformer
import time


def get_loader(config):


    transform_train = torchvision.transforms.Compose([
                      torchvision.transforms.RandomResizedCrop((config.img_size, config.img_size), scale=(0.05, 1.0)),
                      torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_test = torchvision.transforms.Compose([
                     torchvision.transforms.Resize((config.img_size, config.img_size)),
                     torchvision.transforms.ToTensor(),
                     torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if config.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=r'../data',
                                                train=True,
                                                download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=r'../data',
                                               train=False,
                                               download=True,
                                               transform=transform_test)
        print(f'Current Data Set:{config.dataset}')
    else:

        trainset = torchvision.datasets.CIFAR100(root='./data',
                                                 train=True,
                                                 download=True,
                                                 transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data',
                                                train=False,
                                                download=True,
                                                transform=transform_test)
        print('Current Data Set: cifar100')



    print("train number:", len(trainset))
    print("test number:", len(testset))

    train_loader = DataLoader(trainset, batch_size=config.train_batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=config.eval_batch_size, shuffle=False)
    print("train_loader:", len(train_loader))
    print("test_loader:", len(test_loader))

    return train_loader, test_loader




def save_model(config, model,epoch_index):

    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(config.output_dir, "epoch%s_checkpoint.bin" % epoch_index)
    torch.save(model_to_save.state_dict(), model_checkpoint)


#instantiate model
def getVisionTransformers_model(config):

    num_classes = 10 if config.dataset == 'cifar10' else 100
    model = VisionTransformer(config, num_classes, zero_head=True)
    model.to(config.device)
    return model


# eval
def eval(config, model, test_loader):

    eval_loss = 0
    total_acc = 0

    model.eval()
    loss_function = nn.CrossEntropyLoss()
    for i,batch in enumerate(test_loader):
        batch = tuple(t.to(config.device) for t in batch)
        x, y = batch
        #print(y)
        with torch.no_grad():
            logits, _ = model(x) #(bs,num_classes),weight
            batch_loss = loss_function(logits, y)
            #
            eval_loss += batch_loss.item()
            _, preds = logits.max(1)
            #preds = logits.max(1)

            num_correct = (preds == y).sum().item()
            total_acc += num_correct

    loss = eval_loss/len(test_loader)
    acc = total_acc/(len(test_loader)*config.eval_batch_size)
    return loss, acc


def train(config, model):

    print("load dataset.........................")
    #
    train_loader, test_loader = get_loader(config)
    # Prepare optimizer and scheduler
    #optimizer = torch.optim.SGD(model.parameters(),lr=config.learning_rate,momentum=0.9,weight_decay=config.weight_decay)

    optimizer = torch.optim.Adam(model.parameters())

    print("training.........................")
    #
    val_loss_list = []
    val_acc_list = []
    #
    train_loss_list = []
    loss_func = nn.CrossEntropyLoss()
    for i in range(config.total_epoch):
        print('---------------------------------')
        s = time.time()
        print(f'Epoch {i+1} start at {s}.')

        model.train()
        train_loss = 0
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(config.device) for t in batch)
            x, y = batch

            # print(pred_y)
            # print(y)
            loss = model(x,y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #
        train_loss = train_loss/len(train_loader)
        train_loss_list.append(train_loss)
        np.savetxt("train_loss_list.txt", train_loss_list)
        print("Train Epoch:{},loss:{}".format(i+1, train_loss))

        save_model(config, model, i)

        eval_loss, eval_acc = eval(config, model, test_loader)

        val_loss_list.append(eval_loss)
        val_acc_list.append(eval_acc)
        np.savetxt("Eval_loss_list.txt", val_loss_list)
        np.savetxt("Eval_acc_list.txt", val_acc_list)
        print("Eval Epoch:{}\nEval_loss:{}\nEval_acc:{}".format(i+1, eval_loss, eval_acc))
        e = time.time()
        print(f'Epoch {i+1} ended at {e}. Totally {e-s}\'s were used in this Epoch.')

def main():
    config = get_config()
    model = getVisionTransformers_model(config)
    train(config, model)


