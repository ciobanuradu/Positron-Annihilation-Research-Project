# imports

import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# constants

WIDTH = 2 #mm
RADIUS = 600 #mm
ELECTRON_MASS = 0.511 #MeV

# reading data

dataset = pd.read_excel("KR.xlsx", header=1, )

# remove last column

dataset = dataset.iloc[:, :-1]

# convert from angles to radians

dataset.iloc[:, 1] = dataset.iloc[:, 1] * np.pi / 180 

# plot dataset

#plt.hist(dataset.iloc[:, 1], range=(-0.05, 0.05), bins=50)
plt.plot(dataset.iloc[:, 1], dataset.iloc[:, 2], ".k")
plt.show()

# convolution

slit_angle = WIDTH / RADIUS # small angle approximation -> sin(x) ~= x
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

def gaussian_part(x, sigma, mean): 
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) / sigma)**2 / 2) 

def quadratic_part(x, p_f):
    return np.maximum(0.0, (p_f**2 - x**2) * (3 / (4 * p_f ** 3)))

def fit_function(x, mean, sigma, p_f, weight):
    return weight * gaussian_part(x, sigma, mean) + (1- weight) * quadratic_part(x, p_f)



np.convolve([triangle_function(x) for x in x], [fit_function(x) for x in x], mode="same")


# dataset normalization
normalized_dataset = dataset

total_count = np.sum(normalized_dataset.iloc[:, 2])
bin_width = (normalized_dataset.iloc[-1, 1] - normalized_dataset.iloc[0, 1])/len(normalized_dataset.iloc[:, 1])
normalization_factor = total_count * bin_width
normalized_dataset.iloc[:, 2] = normalized_dataset.iloc[:, 2] / normalization_factor

print("total count: ", total_count)
print("bin width: ", bin_width)
print("normalization factor: ", normalization_factor)

# reality check
# print("sum of normalized counts: ", np.sum(normalized_dataset.iloc[:, 2]) * bin_width)

# deconvolve triangle function
# filter_function = triangle_function[triangle_function>0]
# deconvoluted_dataset, _ = np.deconvolve(normalized_dataset.iloc[:, 2], filter_function, mode="same")[1]  
    
# shift dataset so that mean position is 0
mean_position = np.sum(normalized_dataset.iloc[:, 1] * normalized_dataset.iloc[:, 2]) * bin_width
print("mean position: ", mean_position)
# standard deviation
standard_deviation = np.sqrt(np.sum(normalized_dataset.iloc[:, 2] * (normalized_dataset.iloc[:, 1] - mean_position)**2) * bin_width)
print("standard deviation: ", standard_deviation)

# normalized_dataset.iloc[:, 1] = normalized_dataset.iloc[:, 1] - mean_position


popt, cov = scipy.optimize.curve_fit(fit_function, normalized_dataset.iloc[:, 1], normalized_dataset.iloc[:, 2], bounds = ([-0.04, 0.01, -5, 0], [0.04, 5, 5, 1]), p0=[0, 1, 0.5, 0.5])

#print popts with labels
print("mean: ", popt[0])
print("sigma: ", popt[1])
print("p_f: ", popt[2])
print("weight: ", popt[3])

# popt[0] = 0

plt.plot(x, fit_function(x, *popt), "-r")
plt.plot(x, gaussian_part(x, popt[1], popt[0]), "--b")
plt.plot(x, quadratic_part(x, popt[2]), "--g")
plt.plot(normalized_dataset.iloc[:, 1], normalized_dataset.iloc[:, 2], ".k")
plt.show()

# second convolution

convoluted_function = np.convolve(triangle_function, fit_function(x,*popt), mode="same")
plt.plot(x, convoluted_function, ".k")
plt.show()


# fitting curve

# plotting

# processing fit angle into momentum

fermi_angle = popt[2]
# fermi_momentum = 

# processing momentum into energy

fermi_energy = fermi_momentum ** 2 / (2 * ELECTRON_MASS)