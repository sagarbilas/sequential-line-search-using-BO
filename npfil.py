import pySequentialLineSearch
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from joblib import Parallel, delayed
from typing import Optional, Tuple


data = np.load('acquisition-func-comparison.npy')
print(data)
