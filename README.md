# taichi-internal

## How to download CLINC150 dataset
1. clone the data repository
    a. HTTPS
    ```
    git clone https://github.com/clinc/oos-eval.git
    ```
    b. SSH
    ```
    git clone git@github.com:clinc/oos-eval.git
    ```
2. download repo as zip
```
https://github.com/clinc/oos-eval
```
- you should find the data in json format within the `/data` folder of the repository


## How to Use Current Version
```python
import taichi
from taichi import uslp

# set up configurations for training and evaluation
# hyper-parameter
# pre-trained models
uslp = uslp.USLP() # optionally can pass custom path to the config file in the argument

# load pre-trained models
# process data
uslp.init()

# start training pipeline
uslp.train()

# start evaluation pipeline
uslp.eval()
```

#### Note: 
Please make sure to change the paths and parameters to the config file accordingly
