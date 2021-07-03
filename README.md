# Image match system : pytorch

This is a system to calculate image pairs similarity score.

Based on Siamese network, pytorch and Contrastive loss.

## How to use
### 1. Split data
In this step, Dataset will split to train and validation
```python
python3 split_data.py --dataPath "your dataset path" --train_rate 0.8
```
### 2. Train model
In this step, we will train and record the training process
```python
python3 train.py --datas_path "your train dataset path" --model_path "where you save models" --gpu_ids "depends on your gpu"
```

### 3. Test model
```python
python3 test.py --validate_path "your validate dataset path" --model_path "where model is" 
```

### 4. Show train log
```python
python3 train2img.py --validate_path "your validate dataset path" --model_path "where model is" 
```

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
