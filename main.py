import numpy as np
from functions import QuantumWell
import time
import matplotlib.pyplot as plt



W_L = np.arange(1,10.1,0.1) #nm #barrier length
W_L = 5 #nm #only one value used



Qw = QuantumWell(W_L, plot=True)
Qw.normalize('c')
Qw.plot_W_L()

