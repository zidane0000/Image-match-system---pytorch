import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms
from mydataset import DatasetLoad, DatasetGenerate
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import SiameseNetwork, ContrastiveLoss
import time
import numpy as np
import gflags
import sys
from collections import deque
import os
from test import evaluation

def PrintAndWrite(line):
	path = 'train_log.txt'
	with open(path, 'a') as f:
		f.write(line)

if __name__ == '__main__':	
    Flags = gflags.FLAGS
    gflags.DEFINE_bool("cuda", True, "use cuda")
    gflags.DEFINE_bool("save_best", True, "save model if best test loss.")
    gflags.DEFINE_bool("show_net", False, "show net structure")
    gflags.DEFINE_string("datas_path", None, "data folder")
    gflags.DEFINE_float("train_rate", 0.7, "training rate")
    gflags.DEFINE_float("lr", 0.0001, "learning rate")
    gflags.DEFINE_integer("epochs", 2, "number of train epochs")
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size", 128, "number of batch size")
    gflags.DEFINE_integer("every_iter", 500, "show, test, save model every iter")
    gflags.DEFINE_string("model_path", None, "path to store model, if none then don't save")
    gflags.DEFINE_string("gpu_ids", "0,1,2,3", "gpu ids used to train")

    Flags(sys.argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = Flags.gpu_ids
    print("use gpu:", Flags.gpu_ids, "to train.")
    print("Learning rate: ", Flags.lr)

    # Load datas and split to train and test
    AllDataSet = DatasetLoad(Flags.datas_path, train_rate = Flags.train_rate)
    train_datas, test_datas, num_classes = AllDataSet.dataWithSplit()
    train_dataGenerate = DatasetGenerate(train_datas, num_classes, transform=transforms.ToTensor())
    test_dataGenerate = DatasetGenerate(test_datas, num_classes, transform=transforms.ToTensor(), isTest=True)
    trainLoader = DataLoader(train_dataGenerate, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)
    testLoader = DataLoader(test_dataGenerate, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)

    loss_fn = ContrastiveLoss(margin=1)
    net = SiameseNetwork('pretrain_resnet152')

    #summary net
    if Flags.show_net:
        print(net)

    # multi gpu
    if len(Flags.gpu_ids.split(",")) > 1:
        net = torch.nn.DataParallel(net)

    if Flags.cuda:
        net.cuda()

    net.train()

    # Init optimizer and schedular
    optimizer = torch.optim.Adam(net.parameters(),lr = Flags.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience=2, verbose=True)
    optimizer.zero_grad()

    best_loss_val = sys.maxsize
    loss_val = 0
    time_start = time.time()

    for epoch in range(Flags.epochs):
        print('*'*70)
        print('Epoch[%d]:'%(epoch))
        for iteration_id, (img1, img2, label) in enumerate(trainLoader, 1): # 下標從1開始
            if Flags.cuda:
                img1, img2, label = img1.to(device='cuda', dtype=torch.float), img2.to(device='cuda', dtype=torch.float), label.to(device='cuda', dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            predict = net(img1, img2)
            loss = loss_fn(predict, label)
            loss_val += loss.item()
            loss.backward()
            optimizer.step()

            if iteration_id % Flags.every_iter == 0 :
				# Show
				loss_val = loss_val/Flags.every_iter
                PrintAndWrite('[%d]\tloss: %.6f\ttime lapsed: %.2f s'%(iteration_id, loss_val, time.time() - time_start))
                loss_val = 0
                time_start = time.time()

				# ********** Test **********
                test_loss_val = 0.0
                test_acc_val = 0.0
                test_time_start = time.time()

                all_predict = []
                all_testlabel = []

                with torch.no_grad():
                    for _, (test1, test2, testlabel) in enumerate(testLoader, 1):
                        if Flags.cuda:
                            test1, test2, testlabel = test1.to(device='cuda', dtype=torch.float), test2.to(device='cuda', dtype=torch.float), testlabel.to(device='cuda', dtype=torch.float)
                        predict = net(test1, test2)

                        if len(all_predict) == 0:
                            all_predict = predict.cpu().numpy()
                            all_testlabel = testlabel.cpu().numpy()
                        else:
                            all_predict = np.concatenate((all_predict, predict.cpu().numpy()))
                            all_testlabel = np.concatenate((all_testlabel, testlabel.cpu().numpy()))

                        test_loss = loss_fn(predict, testlabel)
                        test_loss_val += test_loss.item()
                        
                        torch_abs = torch.zeros(predict.shape, dtype=predict.dtype, device=predict.device)

                        for i in range(len(predict)):
                            torch_abs[i] = torch.abs(predict[i] - testlabel[i])

                        test_acc = torch.mean(torch_abs)
                        test_acc_val += test_acc
                far, frr, threshold, eer = evaluation(all_predict, all_testlabel, need_print=False)
                test_loss_val = test_loss_val/len(testLoader)
                test_acc_val = 1 - test_acc_val/len(testLoader)
                PrintAndWrite('[%d]\tTest loss: %f\tTest acc : %f\t EER : %f'%(iteration_id, test_loss_val, test_acc_val, far[eer]))

                # change learning rate if loss up
                scheduler.step(test_loss_val)
				# ********** Test **********
				                
                # Save
                if best_loss_val > test_loss_val:
					best_loss_val = test_loss_val
					torch.save(net.state_dict(), Flags.model_path + '/model-inter-' + str(iteration_id) + '-' + "{:.6f}".format(best_loss_val) + ".pt")

                time_start = test_time_start
        print('*'*70)