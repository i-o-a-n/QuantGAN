# QuantGAN

In QuantGAN implementation.ipynb you can find my implementation of the model, described in [Wiese et al., Quant GANs: Deep Generation of Financial Time Series, 2019](https://arxiv.org/abs/1907.06673)

I've used S&P 500 Close Price log returnrs [1st May 2009 : 31st Dec 2018] as in the initial paper. You can load data via code from .ipynb notebook or download my sp500.csv file.

# Env
I suggest trying to reproduce my results in [Colab](https://colab.google/)
All the requirements can be found in requirements.txt
- Here is how you can install all packages
```
!pip freeze > requirements.txt
```

# Results
Model is presented in Python with the help of PyTorch module.
Model configuration can be found in configuration.py
- Here is how you can get necessary classes
```
from configuration import *
```

## Core parts:
* SP500Dataset
* TemporalBlock
* TCN
* Generator
* Discriminator
