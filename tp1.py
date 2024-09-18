import numpy as np
import matplotlib.pyplot as plt

#Defining the factors of the random variables
LENGHT = 100
K = 100
mean = 1
sigma = 1
a = 1
b= 1
A0=2
lambda0=1

#This here is the white noise variable, defined in the first session TP
def white_nosie(number_var):
    return np.random.normal(mean,sigma, number_var)

##This here is the second random Variable, defined in the first session TP
##X_t = a +bZ_t-1 + Z_t
def sum_white_noise(number_variable):
    X = white_nosie(number_variable)
    X_rounded = np.roll(X,1)
    X_rounded[0] = 0
    return a + b*X + X_rounded

#This one computes the sum of the random variables multiplied by 2 to the power of the variable's indexc
def geometric_white_noise(number_variables):
    X = white_nosie(number_variables+K) # generate a white noise 
    summed_variables = np.zeros(number_variables) # generat array of zeros with the same size of random variable
    # Sum on all of the number of random varibales
    for j in range(number_variables):
        #This index is exactly the random variable defines in the last TP's question
        summed_variables[j] = np.sum((2**-j)*X[j-k+K] for k in range(K+1))
    return summed_variables + a 


##This one is to compute the Random Variable with the cos and the random.uniforme variable on 0 and 2pi
def cos_noise(number_variables):
    T=np.arange(0,number_variables) # An array of the number of random variable
    return A0*np.cos(lambda0*T+np.random.uniform(0,2*np.pi,number_variables))+np.random.normal(mean,sigma,number_variables) # returns the cos function

#This function computes the imperical mean of the random variable
def empirical_mean(X):
    assert(len(X) != 0)
    return np.mean(X)

#This function computes the empirical autocovariance of The random variable
def empirical_autocovariance(X,taus,mean=0):
    X_sum = np.zeros(len(taus))
    for i, tau in enumerate(taus):
        #Roll shifts the indexes to be have a difference of indexes = tau
        shifted_X = np.roll(X, tau)
        X_sum_tau = (X-mean) * (shifted_X - mean)
        X_sum[i] = np.mean(X_sum_tau)
    return X_sum    

#These here are the random variables defined in the first TP
X_sum_WN = sum_white_noise(LENGHT)
X_sum_geometric_WN = geometric_white_noise(number_variables=LENGHT)
X_cos_WN = cos_noise(number_variables=LENGHT)
X_WN = white_nosie(LENGHT)

#empirical means of all the previous random variables and indexes of x axis
indexes = np.arange(LENGHT)
empirical_mean_WN = empirical_mean(X_sum_geometric_WN)
empirical_autocovariance_WN = empirical_autocovariance(X_WN,taus=indexes,mean=0)
empirical_autocovariance_sum = empirical_autocovariance(X_sum_WN,mean=1, taus=indexes)
empirical_autocovariance_geo_sum = empirical_autocovariance(X_sum_geometric_WN,mean=1,taus=indexes)
empirical_autocovariance_cos = empirical_autocovariance(X_cos_WN,mean=0,taus=indexes)



#plotting the empirical mean of various random variables 
plt.grid()
plt.plot(indexes,X_sum_geometric_WN, label='rv path', marker='H')
plt.plot(indexes,np.full_like(indexes,empirical_mean_WN), label='empirical mean')
plt.scatter(indexes,empirical_autocovariance_WN, label='empirical autocov', color='green' , marker = 'x')
plt.legend()
plt.show()
