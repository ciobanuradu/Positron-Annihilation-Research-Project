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

# normalized block function from -angle to angle

def block_function(x: float):
    if np.abs(x) <= slit_angle:
        return 1 / (2 * slit_angle)
    else:
        return 0


# x = dataset.iloc[:, 1].to_numpy()
# print("sum of block function: ", np.sum(block_function(x) * (x[1] - x[0])))

#convolute function with itself:
x = dataset.iloc[:, 1].to_numpy()
triangle_function = np.convolve([block_function(x) for x in x], [block_function(x) for x in x], mode="same")

# normalize triangle function
dx = (np.abs(dataset.iloc[:, 1].to_numpy()[-1] - dataset.iloc[:, 1].to_numpy()[0])) / len(dataset.iloc[:, 1].to_numpy())
normalization_factor = np.sum(triangle_function) * dx
triangle_function_normed = triangle_function / normalization_factor 

# triangle_function_normed = triangle_function_normed * 0.1
plt.plot(dataset.iloc[:, 1].to_numpy(), triangle_function_normed)
plt.show()


#check if triangle is normalized
print("sum of triangle function: ", np.sum(triangle_function_normed * (x[1] - x[0])))


# fit function

def gaussian_part(x, sigma, mean): 
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) / sigma)**2 / 2) 

def quadratic_part(x, p_f, mean):
    return np.maximum(0.0, (p_f**2 - (x)**2) * (3 / (4 * p_f ** 3)))

def fit_function(x, mean, sigma, p_f, weight):
    return weight * gaussian_part(x, sigma, mean) + (1- weight) * quadratic_part(x, p_f, mean)

# use this one
def fit_function_2(x, mean, sigma, p_f, weight):
    res = np.convolve(triangle_function_normed, weight * gaussian_part(x, sigma, mean) + (1- weight) * quadratic_part(x, p_f, mean), mode='same')
    return res / (np.sum(res) * dx)


# convolved_fit_function = np.convolve(triangle_function, [fit_function(x) for x in x], mode="same")


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
    

# mean position
mean_position = np.sum(normalized_dataset.iloc[:, 1] * normalized_dataset.iloc[:, 2]) * bin_width
print("mean position: ", mean_position)


# standard deviation
standard_deviation = np.sqrt(np.sum(normalized_dataset.iloc[:, 2] * (normalized_dataset.iloc[:, 1] - mean_position)**2) * bin_width)
print("standard deviation: ", standard_deviation)


# shift dataset so that mean position is 0
# normalized_dataset.iloc[:, 1] = normalized_dataset.iloc[:, 1] - mean_position


# fit curve

popt, cov = scipy.optimize.curve_fit(fit_function_2, 
        normalized_dataset.iloc[:, 1], 
        normalized_dataset.iloc[:, 2], 
        bounds = ([-0.04, 0.001, 0, 0], [0.04, 0.5, 0.05, 1]), 
        p0=[0, 0.04, 0.01, 0.5])


#print popts with labels
print("mean: ", popt[0])
print("sigma: ", popt[1])
print("p_f: ", popt[2])
print("weight: ", popt[3])

# popt[0] = 0

plt.plot(x, fit_function_2(x, *popt), "-y")
plt.plot(x, fit_function(x, *popt), "-r")
plt.plot(x, gaussian_part(x, popt[1], popt[0]), "--b")
plt.plot(x, quadratic_part(x, popt[2], popt[0]), "--g")
plt.plot(normalized_dataset.iloc[:, 1], normalized_dataset.iloc[:, 2], ".k")
plt.show()


# convolution plots

normalized_convoluted_quadratic = np.convolve(triangle_function_normed, quadratic_part(x, popt[2], popt[0]), mode="same")
normalized_convoluted_quadratic = normalized_convoluted_quadratic / (np.sum(normalized_convoluted_quadratic) * dx)
plt.plot(x, normalized_convoluted_quadratic, "--g")

plt.plot(x, fit_function_2(x, *popt), "-r")
plt.plot(normalized_dataset.iloc[:, 1], normalized_dataset.iloc[:, 2], ".k")
plt.show()

convoluted_function = np.convolve(triangle_function, fit_function(x,*popt), mode="same")
plt.plot(x, convoluted_function, ".k")
plt.show()


# processing fit angle into momentum and energy

fermi_angle = popt[2]
#E = MC^2 
p_paralel = np.sqrt(2 * ELECTRON_MASS * ELECTRON_MASS) #(i think?) MeV / c
p_perp = np.tan(fermi_angle) * p_paralel 
fermi_energy = (p_perp ** 2) + (p_paralel ** 2) / (2 * ELECTRON_MASS)

# uncertainty in energy
fermi_angle_uncertainty = np.sqrt(cov[2, 2])
print("fermi angle uncertainty: ", fermi_angle_uncertainty)
print("fermi angle: ", fermi_angle)
p_paralel_uncertainty = np.sqrt(cov[2, 2]) * p_paralel / np.sqrt(fermi_angle)
print("p paralel uncertainty: ", p_paralel_uncertainty)
print("p perp: ", p_perp)
p_perp_uncertainty = np.sqrt(cov[2, 2]) * p_perp / np.sqrt(fermi_angle)
print("p perp uncertainty: ", p_perp_uncertainty)
print("p paralel: ", p_paralel)
fermi_energy_uncertainty = np.sqrt(cov[2, 2]) * fermi_energy / np.sqrt(fermi_angle)
print("fermi energy uncertainty: ", fermi_energy_uncertainty)
print("fermi energy: ", fermi_energy)
print("energy delta:", fermi_energy - ELECTRON_MASS)