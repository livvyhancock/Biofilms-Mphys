# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:44:55 2023

@author: User-PC
"""
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import matplotlib.pyplot as plt

FILE_NAME = r"/Users/livvy/Documents/Mphys/code and data/30072023_kch2D0.2VEISPOT_#1.txt" # the 0.1 and 0.2 v have 2D0 not D0
#FILE_NAME = r"C:\Users\User-PC\Downloads\Uni\Mphys\Kch data\22072023_B4EISPOT_#1.txt"
DATA = np.zeros((0, 11)) # Appends data to empty array
data_type = []

start_line = 57  # Change this to the desired line number

with open(FILE_NAME, 'r') as file:
    for line_number, line in enumerate(file):
        if line_number < start_line:
            continue 

INPUT_FILE = open(FILE_NAME, "r")

for line in INPUT_FILE:
    split_up = line.split() # Can be spaces, if you wish.
    try:
        temp = np.array([float(x) for x in split_up])
        DATA = np.vstack((DATA, temp))
    except ValueError:
        pass

INPUT_FILE.close()

freq_data = DATA[:, 2]
reZ_data = DATA[:, 3]
imZ_data = DATA[:, 4]
#Zmod_data = DATA[:, 6]
measured_data = np.concatenate((reZ_data, imZ_data))
#measured_data = Zmod_data

def circuit_a (freq, Rx, Rl, Cm, Lx):
    w = 2 * np.pi * freq
    Z = 1/((1j*w*Cm) + 1/Rx + 1/(Rx + 1j*w*Lx) + 1/Rl)
    reZ = Z.real
    imZ = Z.imag
    return(reZ, imZ)


def circuit_c (freq, Cm, Rq, Rn, RKch, Rm, Ln, Lm):
    w = 2 * np.pi * freq
    Z = 1/(((1j*w*Cm)+(1/Rq)+(1/(Rn + 1j*w*Ln))+(1/RKch)+(1/(Rm +1j*w*Lm))))
    reZ = Z.real
    imZ = Z.imag
    return(reZ,imZ)



def memristor(freq, Ra, Rb, Cm, La):
    w = 2 * np.pi * freq
    Z = 1/((1/Rb + 1j*Cm*w + 1/(Ra +1j*La*w)))
    reZ = Z.real
    imZ = Z.imag
    return(reZ, imZ)
    #Zmod = np.sqrt(reZ**2 + imZ**2)
    #return(Zmod)


def fit_function_memristor(freq, Ra, Rb, Cm, La):
    reZ, imZ = memristor(freq, Ra, Rb, Cm, La)
    #Zmod = memristor(freq, Ra, Rb, Cm, La)
    #return Zmod
    return np.concatenate((reZ, imZ))


def fit_function_circuit_a(freq, Rx, Rl, Cm, Lx):
    reZ, imZ = circuit_a(freq, Rx, Rl, Cm, Lx)
    return np.concatenate((reZ, imZ))


def fit_function_circuit_c(freq, Cm, Rq, Rn, RKch, Rm, Ln, Lm):
    reZ, imZ = circuit_c(freq, Cm, Rq, Rn, RKch, Rm, Ln, Lm)
    return np.concatenate((reZ, imZ))



def findoptimisedparams_memristor(function, xdata, ydata):
    '''
    the curvefit needs to use function that returns only the y data you are using..
    call with: findoptimisedparams(memristor, freq_data, Zmod_data)

    '''
    optimisedparams, cov = curve_fit(function, xdata, ydata)
    fit_Ra, fit_Rb, fit_Cm, fit_La = optimisedparams
    return fit_Ra, fit_Rb, fit_Cm, fit_La

def findoptimisedparams_circuit_a(function, xdata, ydata):
    '''
    the curvefit needs to use function that returns only the y data you are using..
    call with: findoptimisedparams(memristor, freq_data, Zmod_data)

    '''
    optimisedparams, cov = curve_fit(function, xdata, ydata)
    fit_Rx, fit_Rl, fit_Cm, fit_Lx = optimisedparams
    return fit_Rx, fit_Rl, fit_Cm, fit_Lx


def findoptimisedparams_circuit_c(function, xdata, ydata):
    '''
    the curvefit needs to use function that returns only the y data you are using..
    call with: findoptimisedparams(memristor, freq_data, Zmod_data)

    '''
    optimisedparams, cov = curve_fit(function, xdata, ydata)
    fit_Cm, fit_Rq, fit_Rn, fit_RKch, fit_Rm, fit_Ln, fit_Lm = optimisedparams
    return fit_Cm, fit_Rq, fit_Rn, fit_RKch, fit_Rm, fit_Ln, fit_Lm


def plotfit(freq, data1, data2, function):

    plt.figure(4)
    plt.scatter(np.log(freq), data1, label='Measured Real Part', color='blue')
    plt.plot(np.log(freq), (function(freq,Ra,Rb,Cm,La))[0], label='Fitted Real Part', color='red')
    plt.scatter(np.log(freq), data2, label='Measured Imaginary Part', color='green')
    plt.plot(np.log(freq), (function(freq,Ra,Rb,Cm,La))[1], label='Fitted Imaginary Part', color='orange')
    plt.xlabel('log Frequency')
    plt.ylabel('Impedance modulus')
    plt.legend()
    plt.show()
    return

Ra,Rb,Cm,La = findoptimisedparams_memristor(fit_function_memristor, freq_data, measured_data)
cov = curve_fit(fit_function_memristor, freq_data, measured_data)[1]
print(Ra,Rb,Cm,La)
print(cov)
plotfit(freq_data, reZ_data, imZ_data , memristor)
