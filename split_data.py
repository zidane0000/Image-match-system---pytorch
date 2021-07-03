import numpy as np
import os
import shutil
import random
import gflags
import sys
from datetime import date

def create_directorys(path):
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory : %s failed" % path)
    else:
        print("Successfully created the directory : %s" % path)

def copy_file(src, dst):
    try:
        shutil.copyfile(src, dst)
    except:
        print("Copy %s to %s failed"%(src,dst))        

if __name__ == '__main__':
    np.random.seed(0)
	Flags = gflags.FLAGS
    gflags.DEFINE_string("dataPath", None, "data folder")
    gflags.DEFINE_float("train_rate", 0.8, "training rate")
    Flags(sys.argv)
	
    dataPath = Flags.dataPath
    train_rate = Flags.train_rate
	
	if Flags.train is None:
        print("Error! No dataPath")
        sys.exit()

    # detect the current working directory and print it
    current_path = os.getcwd()
    print("The current working directory is %s" % current_path)
    today = date.today().strftime("%Y_%m_%d")
    print("Today's date:", today)

    if os.path.isdir(current_path + '/' + today):
        print("directory exists")
    else:
        for classPath in os.listdir(os.path.join(dataPath)):
            print("Spliting %s" % classPath)
            one_class_datas = []
            for samplePath in os.listdir(os.path.join(dataPath, classPath)):
                # filePath = os.path.join(dataPath, classPath, samplePath)
                one_class_datas.append(samplePath)

            # prepare folders
            train_dataPath = current_path + '/' + today + '/train/' + classPath
            test_dataPath = current_path + '/' + today + '/test/' + classPath
            create_directorys(train_dataPath)
            create_directorys(test_dataPath)

            #split datas
            random.shuffle(one_class_datas)
            split = int(len(one_class_datas) * train_rate)
            train_datas = one_class_datas[:split]
            test_datas = one_class_datas[split:]

            #copy dats
            for train_data in train_datas:
                copy_file(os.path.join(dataPath, classPath, train_data), os.path.join(train_dataPath,train_data))

            for test_data in test_datas:
                copy_file(os.path.join(dataPath, classPath, test_data), os.path.join(test_dataPath,test_data))