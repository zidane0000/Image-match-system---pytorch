# Image match system : pytorch

This is a system to calculate image pairs similarity score.

Based on Siamese network, pytorch and Contrastive loss.

## How to use
### 

## File Description
### model.py
Define Siamese Network, Backbone Network and loss function

### mydataset
Read datas from disk to memory, Define the dataset length, Y is 0 or 1
```python
Folder look likes below
- Dataset
	- Class1
		- *.png/.jpg/...
	- Class2
		- *.png/.jpg/...
```

### split_data.py
Split dataset to train and validation

### test.py
Load model to validate, Remember to be consistent with the training situation

### train.py
Train model

### train2img.py
Convert train log to figure
