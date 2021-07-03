import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image

# Folder look likes below
# - Dataset
#	- Class1
#		- *.png/.jpg/...
#	- Class2
#		- *.png/.jpg/...
# Load Data paths from disk to memory, if need split, split to train and test data
class DatasetLoad():
    def __init__(self, dataPath, train_rate=0.7):
        np.random.seed(0)
        self.datas, self.num_classes = self.loadToMem(dataPath)
        self.train_rate = train_rate

    def loadToMem(self, dataPath):
        print("begin loading datas path to memory")
        datas = {}      # dict
        idx = 0
        for classPath in os.listdir(os.path.join(dataPath)):
            datas[idx] = []
            for samplePath in os.listdir(os.path.join(dataPath, classPath)):
                filePath = os.path.join(dataPath, classPath, samplePath)
                datas[idx].append(filePath)
            idx += 1
        print("finish loading datas path to memory")
        return datas, idx

    def dataWithoutSplit(self):
        return self.datas, self.num_classes

    def dataWithSplit(self):
        print("begin spliting datas path")
        train_data = {}
        test_data = {}

        for index in range(self.num_classes):
            train_data[index] = []
            test_data[index] = []
            random.shuffle(self.datas[index])
            split = int(len(self.datas[index]) * self.train_rate)
            train_data[index] = self.datas[index][:split]
            test_data[index] = self.datas[index][split:]

        print("finish spliting datas path")
        return train_data, test_data, self.num_classes

class DatasetGenerate(Dataset):
    def __init__(self, datas, num_classes, transform=None, ):
        super(DatasetGenerate, self).__init__()
        np.random.seed(0)
        self.transform = transform
        self.datas = datas
        self.num_classes = num_classes

	# Based on how you defined it
	# Case1 : Total file numbers is length
	# Case2 : Total possilbe pairs is length
	# Suppose 1000 fingerprint
	# If Case1, it only train 1000 pairs once epoch 
	# but in Case2, C(1000,2)=499500, will take too long in 1 epoch
	# So you can design your best way to calculate len
    def __len__(self):
        total_len = len(self.datas)
        return  total_len

    def read_image(self, filePath):
        # 1. No Feature normalization
        img = Image.open(filePath).convert('L')

        # 2.In Siamese networks for one shot learning
        # img = Image.open(filePath)
        # img = np.asarray(img)
        # img = img / img.std() - img.mean()

        # 3. Feature normalization
        # img = Image.open(filePath).convert('L')
        # img = np.asarray(img)
        # img = img / 255

        # 4. Read by RGB
        # img = Image.open(filePath).convert('RGB')
        return img

	# In this case, same class pairs will define 0, vice versa
    def __getitem__(self, index):
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
        # get image from different class
        else:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        # read image
        image1 = self.read_image(image1)
        image2 = self.read_image(image2)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))