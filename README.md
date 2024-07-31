# PA-DDI

## Papers
* PA-DDI: Durg-Durg Interaction Predicting with Position Awareness
## 1. Environment setup
This code has been tested on on the workstation with NVIDIA RTX 3090 GPU with 24GB of video memory, Python 3.7, pytorch 1.11.0. Please install related libraries before running this code:

    pip install -r requirements.txt
## 2. Download the datesets:
Download the datasets here and put them into data directory:
[data](https://pan.baidu.com/s/1Vlb0QCcZx0YizAFXEIvjIg ) code: q5kv
Below is an example:
```
├─data
    ├─durgbank
        ├─transductive
            ├─f_0
                ├─train
                ├─val
                ├─test

```
## 3. Process data:
cd PA-DDI dictory folder and execute the following script
```cmd
python ./data_process/data_tansform.py
```


## 4. Train
cd PA-DDI dictory folder and execute the following script for transductive training
```cmd
python transductive_train.py
```
cd PA-DDI dictory folder and execute the following script for inductive training
```cmd
python inductive_train.py
```
Hyperparameters can be modified in the corresponding config file.
    
## 5. Evaluate
Download the pretrained model.

* [models](https://pan.baidu.com/s/11WTLJKugntSLKUtjsYmhOA ) code: i2ng
  
execute the following script
```cmd
python transductive_test.py
python inductive_test.py
```
    
