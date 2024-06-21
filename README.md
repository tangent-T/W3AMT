# Language-guided 3D action feature learning without ground-truth sample class label #

## Introduction
This is the official implementation for the paper **"Language-guided 3D action feature learning without ground-truth sample class label"**, IEEE Transactions on Neural Networks and Learning System



## Installation
 ***
  Install the corresponding dependencies in the `requirement.txt`:

```python
    pip install requirement.txt
 ```   

## Data generation
    first place the depth map of NTU 120 dataset on ../ntu120dataset
```python   
    cd /generate_data
    python generate_NTU.py
    python generate_text.py
```

## Train
```python  
    cd /train_code 
    python transfer_feature_to_token.py
    python train_point_ntu60_point_text.py --round 0
    python ex_point_feature.py --round 0 --if_sup True
    
    python train_point_ntu60_point_text.py --round 1
    python ex_point_feature.py --round 1 --if_sup True

    python train_point_ntu60_point_text.py --round 2
    python ex_point_feature.py --round 2 --if_sup True

```

## Test
```python
    cd /linear_classify
    python linercls.py
```



# W3AMT
