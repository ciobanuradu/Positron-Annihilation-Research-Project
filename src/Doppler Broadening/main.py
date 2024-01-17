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
AM241 = []
with open("AM241.txt") as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"):
        AM241.append(line)
AM241 = AM241[1:]

BA133 = []
with open("BA133.txt") as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"):
        BA133.append(line)
BA133 = BA133[1:]

CO60 = []
with open("CO60.txt") as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"):
        CO60.append(line)
CO60 = CO60[1:]

NA22 = []
with open("NA22.txt") as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"):
        NA22.append(line)
NA22 = NA22[1:]

#convert to numpy arrays
AM241 = np.array(AM241, dtype=np.float64)
BA133 = np.array(BA133, dtype=np.float64)
CO60 = np.array(CO60, dtype=np.float64)
NA22 = np.array(NA22, dtype=np.float64)

#first plot!!!
x = np.arange(1, 4097, 1)

#calibrate
x = calibration(x)
AM241[:, 0] = calibration(AM241[:, 0])
BA133[:, 0] = calibration(BA133[:, 0])
CO60[:, 0] = calibration(CO60[:, 0])
NA22[:, 0] = calibration(NA22[:, 0])

plt.plot(x, AM241[:, 1], "r", label="AM_241")
plt.plot(x, BA133[:, 1], "g", label="BA_133")
plt.plot(x, CO60[:, 1], "b", label="CO_60")
plt.plot(x, NA22[:, 1], "y", label="NA_22")
plt.legend()
plt.xlabel("Energy (KeV)")
plt.ylabel("Counts")
plt.title("Spectra of the radioactive sources")
plt.show()


def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = (max(y) - min(y))/2.0 + min(y)
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

def gaussian(x, mean, sigma, A):
    return A * np.exp(-((x - mean) / sigma) ** 2 / 2)

#labeling peaks
AM_peak_x = x[150:190]

BA_peak_1_x = x[210:250]
BA_peak_2_x = x[800:830]
BA_peak_3_x = x[880:910]
BA_peak_4_x = x[1035:1065]
BA_peak_5_x = x[1115:1145]

CO_peak_1_x = x[3460:3490]
CO_peak_2_x = x[3930:3950]

NA_peak_1_x = x[1490:1520]
NA_peak_2_x = x[3755:3775]
 
#Half max
AM_half = half_max_x(AM_peak_x, AM241[150:190, 1])
BA_half_1 = half_max_x(BA_peak_1_x, BA133[210:250, 1])
BA_half_2 = half_max_x(BA_peak_2_x, BA133[800:830, 1])
BA_half_3 = half_max_x(BA_peak_3_x, BA133[880:910, 1])
BA_half_4 = half_max_x(BA_peak_4_x, BA133[1035:1065, 1])
BA_half_5 = half_max_x(BA_peak_5_x, BA133[1115:1145, 1])
CO_half_1 = half_max_x(CO_peak_1_x, CO60[3460:3490, 1])
CO_half_2 = half_max_x(CO_peak_2_x, CO60[3930:3950, 1])
NA_half_1 = half_max_x(NA_peak_1_x, NA22[1490:1520, 1])
NA_half_2 = half_max_x(NA_peak_2_x, NA22[3755:3775, 1])

#FWHM
AM_FWHM = [(AM_half[0] + AM_half[1])/2, AM_half[1] - AM_half[0]]
BA_FWHM_1 = [(BA_half_1[0] + BA_half_1[1])/2, BA_half_1[1] - BA_half_1[0]]
BA_FWHM_2 = [(BA_half_2[0] + BA_half_2[1])/2, BA_half_2[1] - BA_half_2[0]]
BA_FWHM_3 = [(BA_half_3[0] + BA_half_3[1])/2, BA_half_3[1] - BA_half_3[0]]
BA_FWHM_4 = [(BA_half_4[0] + BA_half_4[1])/2, BA_half_4[1] - BA_half_4[0]]
BA_FWHM_5 = [(BA_half_5[0] + BA_half_5[1])/2, BA_half_5[1] - BA_half_5[0]]
CO_FWHM_1 = [(CO_half_1[0] + CO_half_1[1])/2, CO_half_1[1] - CO_half_1[0]]
CO_FWHM_2 = [(CO_half_2[0] + CO_half_2[1])/2, CO_half_2[1] - CO_half_2[0]]
NA_FWHM_1 = [(NA_half_1[0] + NA_half_1[1])/2, NA_half_1[1] - NA_half_1[0]]
NA_FWHM_2 = [(NA_half_2[0] + NA_half_2[1])/2, NA_half_2[1] - NA_half_2[0]]

#plotting the FWHM
plt.plot(AM_FWHM[0], AM_FWHM[1], "c+", label="AM_241")
plt.plot(BA_FWHM_1[0], BA_FWHM_1[1], "g+", label="BA_133")
plt.plot(BA_FWHM_2[0], BA_FWHM_2[1], "g+")
plt.plot(BA_FWHM_3[0], BA_FWHM_3[1], "g+")
plt.plot(BA_FWHM_4[0], BA_FWHM_4[1], "g+")
plt.plot(BA_FWHM_5[0], BA_FWHM_5[1], "g+")
plt.plot(CO_FWHM_1[0], CO_FWHM_1[1], "b+", label="CO_60")
plt.plot(CO_FWHM_2[0], CO_FWHM_2[1], "b+")
plt.plot(NA_FWHM_1[0], NA_FWHM_1[1], "r+", label="NA_22; 511KeV")
plt.plot(NA_FWHM_2[0], NA_FWHM_2[1], "y+", label="NA_22")
plt.legend()
plt.xlabel("Energy (KeV)")
plt.ylabel("FWHM (KeV)")
plt.show()

#correct for experimental error
def error(x, a, b):
    return a * x + b

peak_positions = [AM_FWHM[0], BA_FWHM_1[0], BA_FWHM_2[0], BA_FWHM_3[0], BA_FWHM_4[0], BA_FWHM_5[0], CO_FWHM_1[0], CO_FWHM_2[0], NA_FWHM_2[0]]
peak_heights = [AM_FWHM[1], BA_FWHM_1[1], BA_FWHM_2[1], BA_FWHM_3[1], BA_FWHM_4[1], BA_FWHM_5[1], CO_FWHM_1[1], CO_FWHM_2[1], NA_FWHM_2[1]]

popt, cov = scipy.optimize.curve_fit(error, peak_positions, peak_heights)

plt.plot(NA_FWHM_1[0], NA_FWHM_1[1], "r+", label="511 KeV peak")
plt.plot(x, error(x, *popt), "--k", label="linear fit")
plt.plot(peak_positions, peak_heights, "g+", label="other peaks")
plt.legend()
plt.xlabel("Energy (KeV)")
plt.ylabel("FWHM (KeV)")
plt.title("FWHM of peaks with error function fit")
plt.show()

# compensate for error
AnnihilationFWHM = np.sqrt(NA_FWHM_1[1] ** 2 - error(NA_FWHM_1[0], *popt) ** 2)
print("AnnihilationFWHM: ", AnnihilationFWHM, " KeV")
momentum = AnnihilationFWHM # KeV/c
print("momentum: ", momentum, " KeV/c")
fermi_energy = momentum ** 2 / (2 * 511) # KeV
print("fermi_energy: ", fermi_energy * 1000, " eV")

#NONSENSE BEGINS HERE:

# fitting gaussian to 511 KeV peak
popt, cov = scipy.optimize.curve_fit(gaussian, NA_peak_1_x, NA22[1490:1520, 1], p0=[511, 5, 1750])

print("sigma: ", popt[0])
print("mean: ", popt[1])
print("A: ", popt[2])

print("standard error ",popt[1] / np.sqrt(np.sum(NA22[1490:1520, 1])))

residuals = NA22[1490:1520, 1] - gaussian(NA_peak_1_x, *popt)
MSE = np.sum(residuals ** 2) / len(residuals)
print("MSE: ", MSE)
plt.plot(NA_peak_1_x, NA22[1490:1520, 1], "r+", label="511 KeV peak")
plt.plot(NA_peak_1_x, gaussian(NA_peak_1_x, *popt), "--k", label="gaussian fit")
plt.show()

sigma = np.sqrt(MSE)
energy_uncertainty = (np.sqrt(cov[1][1]) * 511) ** 2 / (2 * 511) # KeV
print("energy_uncertainty: ", energy_uncertainty * 1000, " eV")
