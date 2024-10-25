import numpy as np
import matplotlib.pyplot as plt

# Initializing the variables here
np.random.seed(0)
NUMBER_VARIABLES = 1000
assert(NUMBER_VARIABLES>4)
phis = np.random.randint(1,1000,size=(1,1004))

def white_noise(mean,sigma,number_variables):
    return np.random.normal(mean,sigma,1_000)

def AR(p):
    AR_process = np.zeros(NUMBER_VARIABLES+4)
    initial_matrix = np.random.randint(1,1000,size=(1,4))
    AR_process[0:3] = initial_matrix[0:3]
    for variable in range(4,NUMBER_VARIABLES+4):
        AR_process[variable] = phis[variable-1]*AR_process[variable-1] 
        + phis[variable-2]*AR_process[variable-2]
        + phis[variable-3]*AR_process[variable-3]
        + phis[variable-4]*AR_process[variable-4]
    return AR_process[4:]

def Gamma(p):
    pass





