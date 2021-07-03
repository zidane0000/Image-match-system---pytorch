import numpy as np
import gflags
import sys
import matplotlib.pyplot as plt
import os
import re

if __name__ == '__main__':    
    Flags = gflags.FLAGS
    gflags.DEFINE_string("train", None, "train log path")
    Flags(sys.argv)
    
    if Flags.train is None:
        print("Error! No train log")
        sys.exit()
    
    lines = []
    with open(Flags.train,'r') as fp:
        lines = fp.readlines()

    if len(lines) == 0:
        print("Error! train log empty")
        sys.exit()

    train_epoch = []
    train_loss = []

    test_epoch = []
    test_loss = []
    test_acc = []
    test_eer = []

    # match pattern
    train_pattern = '\[(\d+)\] +loss: (\d+.\d*)'
    test_pattern = '\[(\d{3,9})\] +Test loss: (\d+.\d*) +Test acc : (\d+.\d*) +EER : (\d+.\d*)'
    for line in lines:
        train_res = re.match(train_pattern, line)
        if train_res:
            train_epoch.append(float(train_res.group(1)))
            train_loss.append(float(train_res.group(2)))

        test_res = re.match(test_pattern, line)
        if test_res:
            test_epoch.append(float(test_res.group(1)))
            test_loss.append(float(test_res.group(2)))
            test_acc.append(float(test_res.group(3)))
            test_eer.append(float(test_res.group(4)))

    best_model_epoch = test_epoch[test_loss.index(min(test_loss))]

    
    fig, axs = plt.subplots(3,figsize=(5,8))

    axs[0].plot(train_epoch, train_loss)     # plot 損失函數變化曲線
    axs[0].plot(test_epoch, test_loss)     # plot 損失函數變化曲線
    axs[0].legend(['train_loss','test_loss'], loc='upper left')
    axs[0].axvline(x=best_model_epoch, linestyle= '--')

    axs[1].plot(test_epoch, test_acc)     # plot 損失函數變化曲線
    axs[1].legend(['test_acc'], loc='upper left')
    axs[1].axvline(x=best_model_epoch, linestyle= '--')

    axs[2].plot(test_epoch, test_eer)     # plot 損失函數變化曲線
    axs[2].legend(['test_eer'], loc='upper left')
    axs[2].axvline(x=best_model_epoch, linestyle= '--')

    plt.xlabel('epoch')
    plt.show()