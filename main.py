import numpy as np
from functions import QuantumWell
import time
import matplotlib.pyplot as plt



aW = np.arange(1,10.1,0.1) #nm #barrier length
aW = 5 #nm #only one value used



Qw = QuantumWell(aW, plot=True)
Qw.normalize('c')
Qw.plot_aW()

