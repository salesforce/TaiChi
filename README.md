# taichi-internal
## High Level Summary

**Why are we doing this?**

1. Large, complex and isolated codebases from existing few shot learning projects (research+product) → it’s a BIG learning curve for new users!
2. Open source libraries for few shot learning are extremely rare...
3. A library with unified, user-friendly and pythonic APIs is desired to empower any Python user to quickly experiment with their own data

**Enter Taichi - An open source Python library for few shot learning**

1. Modular and extensible API design, “*from taichi import few_shot_learning_method*”
2. Two few shot methods have been implemented, USLP + DNNC
3. Supports quick data sampling and error analysis

## High Level Details on Algorithms



## How to download CLINC150 dataset
- clone the data repository


    a. HTTPS
    ```
    git clone https://github.com/clinc/oos-eval.git
    ```
    b. SSH
    ```
    git clone git@github.com:clinc/oos-eval.git
    ```
- You should find the data in json format within the `/data` folder of the repository
- Please refer to the [readme](https://github.com/clinc/oos-eval/blob/master/README.md) for more information on the dataset


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
