# imports
import scipy.optimize, scipy.signal, scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# constants

WIDTH = 2 #mm
RADIUS = 600 #mm
ELECTRON_MASS = 510.9989461 #keV

# reading data

dataset = pd.read_excel("KR.xlsx", header=1, )

# remove last column

dataset = dataset.iloc[:, :-1]
dataset = dataset.to_numpy()

#convert 2nd column to radians
dataset[:, 1] = dataset[:, 1] * np.pi / 180

# normalize by first column
dataset[:, 2] = dataset[:, 2] / dataset[:, 0]

normalized_dataset = dataset

total_count = np.sum(normalized_dataset[:, 2])
bin_width = (normalized_dataset[-1, 1] - normalized_dataset[0, 1])/len(normalized_dataset[:, 1])
normalization_factor = total_count * bin_width
normalized_dataset[:, 2] = normalized_dataset[:, 2] / normalization_factor

# convolution

slit_angle = WIDTH / ( 2 * RADIUS) # small angle approximation -> sin(x) ~= x

# normalized block function from -angle to angle

def block_function(x: float):
    if np.abs(x) <= slit_angle:
        return 1 / (2 * slit_angle)
    else:
        return 0

#convolute function with itself:
x = dataset[:, 1]
triangle_function = np.convolve([block_function(x) for x in x], [block_function(x) for x in x], mode="same")


# normalize triangle function
normalization_factor = max(triangle_function) / max([block_function(i) for i in x]) # peak of triangle function should be 300
triangle_function = triangle_function / normalization_factor 

# fit function

def gaussian_part(x, sigma): 
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-((x) / sigma)**2 / 2) 

def quadratic_part(x, p_f):
    return np.maximum(0.0, (p_f**2 - (x)**2) * (3 / (4 * p_f ** 3)))

dx = (np.abs(dataset[-1, 1] - dataset[0, 1])) / len(dataset[:, 1])

def fit_function(x, sigma, p_f, weight, bias, background):
    x = x - bias
    res = np.convolve(triangle_function, weight * gaussian_part(x, sigma) + (1- weight) * quadratic_part(x, p_f), mode='same') + background
    return res / (np.sum(res) * dx)

popt, cov = scipy.optimize.curve_fit(fit_function, 
        normalized_dataset[:, 1], 
        normalized_dataset[:, 2], 
        bounds = ([0.001, 0, 0, -0.1, 0], [0.5, 0.05, 1, 0.1, 1000]), 
        p0=[0.01, 0.01, 0.5, 0, 0])

fitted_dataset = fit_function(normalized_dataset[:, 1], *popt)

plt.plot(normalized_dataset[:, 1], normalized_dataset[:, 2], ".k")
plt.plot(normalized_dataset[:, 1], fit_function(normalized_dataset[:, 1], *popt), "b")
plt.show()

print("sigma: ", popt[0], " p_f: ", popt[1], " weight: ", popt[2], " bias: ", popt[3], " background: ", popt[4])
# processing fit angle into momentum and energy

fermi_angle = popt[1]
#fermi_energy = fermi_angle ** 2 * ELECTRON_MASS / 2 # KeV
fermi_energy = (np.sqrt(fermi_angle ** 2 + 1) - 1) * ELECTRON_MASS # KeV # E^2 = m^2 + p^2

# uncertainty in energy
fermi_angle_uncertainty = np.sqrt(np.sum(cov[1, 1]))
fermi_energy_uncertainty = np.sin(fermi_angle * 2) * ELECTRON_MASS / 2 * fermi_angle_uncertainty 

print("fermi angle uncertainty: ", fermi_angle_uncertainty)
print("fermi angle: ", fermi_angle)

print("fermi energy uncertainty: ", fermi_energy_uncertainty * 1000, " eV")
print("fermi energy: ", fermi_energy * 1000, " eV")
