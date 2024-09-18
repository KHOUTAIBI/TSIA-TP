import numpy as np
import matplotlib.pyplot as plt

#This here is the white noise variable, defined in the first session TP
def white_nosie(mean,sigma , number_var):
    return np.random.normal(mean,sigma, number_var)

##This here is the second random Variable, defined in the first session TP
##X_t = a +bZ_t-1 + Z_t
def sum_white_noise(a,b,number_variable):
    X = white_nosie(0,1,number_variable)
    X_rounded = np.roll(X,1)
    X_rounded[0] = 0
    return a + b*X + X_rounded

#This one computes the sum of the random variables multiplied by 2 to the power of the variable's indexc
def geometric_white_noise(K,a, number_variables):
    X = white_nosie(0,1,number_variables+K) # generate a white noise 
    summed_variables = np.zeros(number_variables) # generat array of zeros with the same size of random variable
    # Sum on all of the number of random varibales
    for j in range(number_variables):
        #This index is exactly the random variable defines in the last TP's question
        summed_variables[j] = np.sum((2**-j)*X[j-k+K] for k in range(K+1))
    return summed_variables + a 


##This one is to compute the Random Variable with the cos and the random.uniforme variable on 0 and 2pi
def cos_noise(A0,lambda0,number_variables):
    T=np.arange(0,number_variables) # An array of the number of random variable
    return A0*np.cos(lambda0*T+np.random.uniform(0,2*np.pi,number_variables))+np.random.normal(0,1,number_variables) # returns the cos function

#This function computes the imperical mean of the random variable
def empirical_mean(X):
    assert(len(X) != 0)
    return np.mean(X)

#This function computes the empirical autocovariance of The random variable
def empirical_autocovariance(X,taus):
    X_sum = np.zeros(len(taus))
    for i, tau in enumerate(taus):
        #Roll shifts the indexes to be have a difference of indexes = tau
        shifted_X = np.roll(X, tau)
        X_sum_tau = X * shifted_X
        X_sum[i] = np.mean(X_sum_tau)
    return X_sum    
    
X_sum_WN = sum_white_noise(1,1,100)
X_sum_geometric_WN = geometric_white_noise(K=50,a=1,number_variables=100)
X_cos_WN = cos_noise(A0=2,lambda0=1,number_variables=100)
X = white_nosie(0,1,100)

#empirical mean and indexes of x axis
indexes = np.arange(100)
empirical_mean_WN = empirical_mean(X_cos_WN)
empirical_autocovariance_WN = empirical_autocovariance(X_sum_geometric_WN,taus=indexes)


#plotting the empirical mean of various random variables 
plt.grid()
plt.plot(indexes,X_cos_WN, label='rv path', marker='H')
plt.plot(indexes,np.full_like(indexes,empirical_mean_WN), label='empirical mean')
plt.plot(indexes,empirical_autocovariance_WN, label='empirical autocov')
plt.legend()
plt.show()
