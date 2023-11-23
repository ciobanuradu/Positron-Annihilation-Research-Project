# imports

import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# constants

WIDTH = 2 #mm
RADIUS = 600 #mm

# reading data

dataset = pd.read_excel("KR.xlsx", header=1, )

# remove last column

dataset = dataset.iloc[:, :-1]

# convert from angles to radians

#dataset.iloc[:, 1] = dataset.iloc[:, 1] * np.pi / 180 

# plot dataset

#plt.hist(dataset.iloc[:, 1], range=(-0.05, 0.05), bins=50)
plt.plot(dataset.iloc[:, 1], dataset.iloc[:, 2], ".k")
plt.show()

# convolution

slit_angle = WIDTH / np.sqrt(RADIUS**2 + WIDTH**2)
print(slit_angle)

# normalized block function from -angle to angle

def block_function(x: float):
    if np.abs(x) <= slit_angle:
        return 1 / (2 * slit_angle)
    else:
        return 0

#convolute function with itself:
x = np.linspace(-1.7, 1.7, 5000)
triangle_function = np.convolve([block_function(x) for x in x], [block_function(x) for x in x], mode="same")
# fit function

def fit_function(x, mean, sigma, p_f, weight):
    gaussian_part = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) / sigma)**2 / 2) 
    quadratic_part = np.maximum(0.0, (p_f**2 - x**2) * (3 / (4 * p_f ** 3)))
    return weight * gaussian_part + (1- weight) * quadratic_part
normalized_dataset = dataset

total_count = np.sum(normalized_dataset.iloc[:, 2])
bin_width = (normalized_dataset.iloc[-1, 1] - normalized_dataset.iloc[0, 1])/len(normalized_dataset.iloc[:, 1])
normalization_factor = total_count * bin_width

print("total count: ", total_count)
print("bin width: ", bin_width)
print("normalization factor: ", normalization_factor)

normalized_dataset.iloc[:, 2] = normalized_dataset.iloc[:, 2] / normalization_factor
# shifting_factor = np.sum([normalized_dataset.iloc[i, 2] * normalized_dataset.iloc[i, 1] for i in range(len(normalized_dataset.iloc[:, 1]))])/np.sum(normalized_dataset.iloc[:, 2])

popt, cov = scipy.optimize.curve_fit(fit_function, normalized_dataset.iloc[:, 1], normalized_dataset.iloc[:, 2], bounds = ([-1, 0.01, -5, 0], [1, 5, 5, 1]), p0=[0, 1, 0.5, 0.5])

#print popts with labels
print("mean: ", popt[0])
print("sigma: ", popt[1])
print("p_f: ", popt[2])
print("weight: ", popt[3])

# popt[0] = 0

plt.plot(x, fit_function(x, *popt), "-r")
plt.plot(normalized_dataset.iloc[:, 1], normalized_dataset.iloc[:, 2], ".k")
plt.show()

# second convolution

convoluted_function = np.convolve(triangle_function, fit_function(x, 0, 0.01, 0.5, 0.5), mode="same")

# fitting curve

# plotting

# processing fit angle into momentum

# processing momentum into energy