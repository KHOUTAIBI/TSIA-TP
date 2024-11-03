import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sg
import scipy.signal as sp

# Initializing the variables here
NUMBER_VARIABLES = 1_000
assert NUMBER_VARIABLES>4 
SIGMA = 4
PHIS = np.array([0.55,-0.7,0.3,0.1])

print(f"the real phis are {PHIS}")

def white_noise(mean,sigma,number_variables):
    return np.random.normal(mean,sigma,number_variables)

def AR(p,phis,number_variables):
    AR_process = np.random.normal(0,SIGMA,size=number_variables)
    for variable in range(number_variables):
        for shift in range(p):
            AR_process[variable] += (phis[shift] * AR_process[variable-shift-1]) 
    return AR_process



def spectral_density(nu,phis):
    w = 2*np.pi*nu
    phis = np.append([1],-phis)    
    nu , Sxx = sp.freqz([1],phis,w)
    return nu , Sxx

def empirical_roots(empirical_phis):
    return np.roots(empirical_phis)

def spectral_density_roots(nu,phis,sigma):    
    roots = empirical_roots(phis)
    expos = 1
    for root in roots:
        expos *= np.abs(1/(np.exp(1j*nu) - root))**2
    return ((sigma**2)/2*np.pi)*expos

def MSE(phis,estimated_phis):
    pass
# The plotting part 

# the gamme, sigma and phis values here
gamma , sigma_approximation , empirical_phis = Gamma(4,PHIS,NUMBER_VARIABLES)

# Plotting time
# Plotting the spectral density using FREQZ

nu = np.linspace(-0.5,0.5,20_000)
empirical_spectral = spectral_density(nu,empirical_phis)
theoretical_spectral = spectral_density(nu,PHIS)
w = 2*np.pi*nu
plt.grid()
plt.plot(w,np.abs(empirical_spectral[1]))
plt.plot(w,np.abs(theoretical_spectral[1]))
plt.xlabel("nu")
plt.ylabel("module")
plt.legend(["empirical power spectral density","theoretical power spectral density"])
plt.show()

# Plotting the roots with the real and empirical coefficients
angles = np.linspace(0,2*np.pi,20_000)
emp_roots = empirical_roots(empirical_phis)
real_roots = np.roots(PHIS)
plt.grid()
plt.plot(np.sin(angles),np.cos(angles),linewidth=2)
plt.scatter(np.real(emp_roots),np.imag(emp_roots),marker='o',color='purple')
plt.scatter(np.real(real_roots),np.imag(real_roots),marker='x',color='red')
plt.legend(["unit cirle","empirical roots","real roots"])
plt.show()


# Spectral density with roots 
nu = np.linspace(-np.pi,np.pi,20_000)
empirical_phis = np.flip(np.append([1],-empirical_phis))
_phi = np.flip(np.append([1],-PHIS))
empirical_density = spectral_density_roots(nu,empirical_phis,sigma_approximation)
theo_density = spectral_density_roots(nu,_phi,SIGMA)


# Plotting the spectral density using the roots method
plt.grid()
plt.plot(nu,empirical_density)
plt.plot(nu,theo_density)
plt.xlabel("nu")
plt.ylabel("module")
plt.legend(["empirical power spectral density","theoretical power spectral density"])
plt.show()