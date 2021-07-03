import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms
from mydataset import DatasetLoad, DatasetGenerate
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import SiameseNetwork, ContrastiveLoss
import time
import numpy as np
import gflags
import sys
import os

# https://medium.com/@mustafaazzurri/face-recognition-system-and-calculating-frr-far-and-eer-for-biometric-system-evaluation-code-2ac2bd4fd2e5
# Evaluate by FAR(False Acceptance Rate), FRR(False Rejection Rate) and EER(Equal Error Rate)
def evaluation(predict, Y, need_print=True):
    # predict : predict value
    # Y : ground truth
    far = []
	frr = []
	eer_posi = float('nan')
	eer_value = float('nan')
	threshold = []	
	for i in np.arange(0.0, 1.1, 0.01):
		# Make sure your ground truth corresponding to the far and frr
		# In this case, same pairs is 0, so false accept is Y==1
		num_far = (np.sum((predict<=i) & (Y == 1)) / np.sum(Y==1)) * 100
		num_frr = (np.sum((predict>=i) & (Y == 0)) / np.sum(Y==0)) * 100

		far.append(num_far)
		frr.append(num_frr)
		threshold.append(i)
		
		if math.isnan(eer_posi) and (num_far == num_frr or num_far < num_frr):
			eer_posi = i
			eer_value = num_far
			print(eer_posi)
			print(eer_value)

	# Show result by figure
	fig, ax = plt.subplots()
	ax.plot(threshold, far, 'r--', label='FAR')
	ax.plot(threshold, frr, 'b--', label='FRR')
	plt.title('FAR and FRR')
	plt.xlabel('Threshold')
	plt.plot( eer_posi, eer_value,'ro', label='EER') 
	plt.axis([0.0, 1.0, 0, 100])
	legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
	plt.show()
    
if __name__ == '__main__':
    
    Flags = gflags.FLAGS
    gflags.DEFINE_bool("cuda", True, "use cuda")
    gflags.DEFINE_string("test_path", None, 'path of testing folder')
    gflags.DEFINE_string("model_path", None, 'path of model')
    gflags.DEFINE_integer("batch_size", 64, "number of batch size")
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_string("gpu_ids", "0,1,2,3", "gpu ids used to train")
    Flags(sys.argv)

    if Flags.test_path == None:
        print("Error, No test path!")
        sys.exit()
    
    if Flags.model_path == None:
        print("Error, No model path!")
        sys.exit()

    test_datas, num_classes = DatasetLoad(Flags.test_path).dataWithoutSplit()
    test_dataGenerate = DatasetGenerate(test_datas, num_classes, transform=transforms.ToTensor(), isTest=True)
    testLoader = DataLoader(test_dataGenerate, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)

    loss_fn = ContrastiveLoss(margin=1)
    net = SiameseNetwork('pretrain_resnet152')

    os.environ["CUDA_VISIBLE_DEVICES"] = Flags.gpu_ids
    print("use gpu:", Flags.gpu_ids, "to test.")

    # multi gpu
    if Flags.cuda and len(Flags.gpu_ids.split(",")) > 1:
        net = torch.nn.DataParallel(net)

    if Flags.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    tensors = torch.load(Flags.model_path, map_location=device)
    net.load_state_dict(tensors)
    net.to(device)
    net.eval()

    test_loss_val = 0
    test_acc_val = 0.0

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

            test_acc = 1 - torch.mean(torch_abs) # need 1- ..
            test_acc_val += test_acc
        
        print('Test loss: %f\tTest acc : %f'%(test_loss_val/len(testLoader), test_acc_val/len(testLoader)))
    
    evaluation(all_predict, all_testlabel)