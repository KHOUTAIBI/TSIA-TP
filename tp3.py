import numpy as np
import matplotlib.pyplot as plt

# Initializing the variables here
np.random.seed(0)
NUMBER_VARIABLES = 1000
assert(NUMBER_VARIABLES>4)
phis = np.random.randint(1,1000,size=(4))
print(phis)

def white_noise(mean,sigma,number_variables):
    return np.random.normal(mean,sigma,1_000)

def AR(p):
    AR_process = np.zeros(NUMBER_VARIABLES+4)
    initial_matrix = np.random.randint(1,1000,size=(1,4))
    AR_process[0:3] = initial_matrix[0:3]
    for variable in range(4,NUMBER_VARIABLES+4):
        AR_process[variable] = phis[0]*AR_process[variable-1] 
        + phis[1]*AR_process[variable-2]
        + phis[3]*AR_process[variable-3]
        + phis[4]*AR_process[variable-4]
    return AR_process[4:]

def Gamma(p):
    phi_matrix = np.dot(np.array(
        [1,-phis[0],-phis[1],-phis[2],-phis[3]]
    ),
    np.transpose(
        np.array([1,-phis[0],-phis[1],-phis[2],-phis[3]])
            )
        
    )

    print(np.array(
        [1,-phis[0],-phis[1],-phis[2],-phis[3]]
    ))
    return phi_matrix

print(Gamma(4))



