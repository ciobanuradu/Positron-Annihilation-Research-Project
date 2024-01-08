#imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize, scipy.signal
import csv

#constants

#calibration
def calibration(x):
    return 0.3379 * x + 1.82439 # KeV

#reading data