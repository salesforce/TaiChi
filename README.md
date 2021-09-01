# taichi-internal

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
