import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sg
import scipy.signal as sp

# Initializing the variables here
NUMBER_VARIABLES = 1_000
assert NUMBER_VARIABLES>4 
SIGMA = 2
PHIS = np.array([0.1,0.2,0.15,0.4])
print(f"phis are {PHIS}")

def white_noise(mean,sigma,number_variables):
    return np.random.normal(mean,sigma,number_variables)

def AR(p):

    AR_process = np.random.normal(0,SIGMA,size=NUMBER_VARIABLES)
    
    for variable in range(NUMBER_VARIABLES):
        for shift in range(p):
            AR_process[variable] += (PHIS[shift] * AR_process[variable-shift-1]) 
    return AR_process

def empirical_mean(X):
    assert(len(X) != 0)
    return np.mean(X)

def empirical_autocovariance(X, taus):
    N = len(X)
    X_sum = np.zeros(len(taus))
    
    for tau, k in enumerate(taus):
        X_shifted = X[k:N] - empirical_mean(X)
        X_original = X[0:N-k] - empirical_mean(X)
        X_sum[tau] = (1/N)  * np.sum(X_shifted * X_original)
    
    return X_sum   

def Gamma(p):
    AR_process = AR(p)
    taus = np.arange(0,p+1)
    empirical_autocov = empirical_autocovariance(AR_process,taus=taus) 
    gamma = sg.toeplitz(empirical_autocov)
    # printing the toeplitz values in order to test the validity of the result

    approximation = np.linalg.inv(gamma) @ np.array([1,0,0,0,0]).T
    sigma_approximation = 1/approximation[0]
    empirical_phis = -(sigma_approximation*approximation[1:])
    print(f"empirical sigma is {sigma_approximation}")
    print(f"empirical phis are {empirical_phis}")

    MSE = np.sum((PHIS-empirical_phis)**2/(PHIS**2))
    print(f"the MSE is equal to: {MSE}")
    
    return gamma 

def spectral_density(nu,p):
    AR_process = AR(p)
    gamma = Gamma(p)
    approximation = np.linalg.inv(gamma) @ np.array([1,0,0,0,0]).T
    sigma_approximation = 1/approximation[0]
    empirical_phis = -(sigma_approximation*approximation[1:])
    w = 2*np.pi*nu    
    nu , Sxx = sp.freqz([1],empirical_phis,w)
    return nu , Sxx


def theo_spectral_density(nu):
    w = 2*np.pi*nu    
    nu , Sxx = sp.freqz([1],PHIS,w)
    return nu , Sxx


nu = np.linspace(-0.5,0.5,20_000)
empirical_spectral = spectral_density(nu,4)
theoretical_spectral = theo_spectral_density(nu)

plt.grid()
plt.plot(nu,np.abs(empirical_spectral[1]))
plt.plot(nu,np.abs(theoretical_spectral[1]))
plt.xlabel("nu")
plt.ylabel("module")
plt.legend(["empirical power spectral density","theoretical power spectral density"])

plt.show()
