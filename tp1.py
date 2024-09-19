import numpy as np
import matplotlib.pyplot as plt

#Defining the factors of the random variables
LENGHT = 100
K = 100
mean = 0
sigma = 1
a = 1
b = 1
A0 = 2
lambda0 = 1

#This here is the white noise variable, defined in the first session TP
def white_nosie(number_var):
    return np.random.normal(mean,sigma, number_var)

##This here is the second random Variable, defined in the first session TP
##X_t = a +bZ_t-1 + Z_t
def sum_white_noise(number_variable):
    X = white_nosie(number_variable)
    X_1 = white_nosie(number_var=1)
    X_rounded = np.roll(X,1)
    X_rounded = a + b*X + X_rounded
    X_rounded[0] = a + b*X[0] + X_1[0]  
    return X_rounded

#This one computes the sum of the random variables multiplied by 2 to the power of the variable's indexc
def geometric_white_noise(number_variables):
    X = white_nosie(number_variables+K) # generate a white noise 
    summed_variables = np.zeros(number_variables) # generate array of zeros with the same size of random variable
    # Sum on all of the number of random varibales
    for j in range(number_variables):
        #This index is exactly the random variable defines in the last TP's question

        ##Fixed j => k and it should work now
        summed_variables[j] = np.sum((2**-k)*X[j-k+K] for k in range(K+1))
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
def empirical_autocovariance(X, taus, mean=0):
    N = len(X)
    X_sum = np.zeros(len(taus))
    for tau, k in enumerate(taus):
        X_shifted = X[k:N] - mean
        X_original = X[0:N-k] - mean
        X_sum[tau] = 1/N  * np.sum(X_shifted * X_original)
    return X_sum   

#This function return theoretical autocov of WN
def theoretical_autocov_WN(X):
    autocov = np.zeros_like(X)
    autocov[0] = sigma**2
    return autocov

#This one for the sum of WN
def theoretical_autocov_sum_WN(X):
    autocov = np.zeros_like(X)
    autocov = np.where((autocov == 1) | (autocov == len(X)), b**2 * sigma**2, 0)
    autocov[0] = (1+b**2)*sigma**2
    return autocov

#This one for the geometric sum of WN
def theoretical_autocov_geo_sum_WN(X):
    autocov = np.zeros_like(X)
    for j in range(len(X)):
        if abs(j) <= K:
            autocov[j] = (sigma**2)* (2**-abs(j))*((1-(1/4)**(K-j+1)))/(3/4)
        else:
            autocov[j] = 0
    return autocov

#This one for the autocov of cos
def theoretical_autocov_cos_WN(X):
    autocov = np.zeros_like(X)
    autocov[0] = sigma**2 + (1/2) * (A0**2)
    for j in range(1,len(X)):
        autocov[j] = (1/2) * (A0**2) * (np.cos(lambda0*j))
    return autocov

#These here are the random variables defined in the first TP
X_sum_WN = sum_white_noise(number_variable=LENGHT)
X_sum_geometric_WN = geometric_white_noise(number_variables=LENGHT)
X_cos_WN = cos_noise(number_variables=LENGHT)
X_WN = white_nosie(number_var=LENGHT)

#empirical means of all the previous random variables and indexes of x axis
indexes = np.arange(LENGHT)
empirical_mean_WN = empirical_mean(X_WN)

#Empirical autocov of the random variables 
empirical_autocovariance_WN = empirical_autocovariance(X_WN,taus=indexes,mean=0)
empirical_autocovariance_sum = empirical_autocovariance(X_sum_WN,mean=1, taus=indexes)
empirical_autocovariance_geo_sum = empirical_autocovariance(X_sum_geometric_WN,mean=1,taus=indexes)
empirical_autocovariance_cos = empirical_autocovariance(X_cos_WN,mean=0,taus=indexes)

#Theoretical autocov of the random variables
theoretical_autocovariance_WN = theoretical_autocov_WN(X_WN)
theoretical_autocovariance_sum = theoretical_autocov_sum_WN(X_sum_WN)
theoretical_autocovariance_geo_sum = theoretical_autocov_geo_sum_WN(X_sum_geometric_WN)
theoretical_autocovariance_cos = theoretical_autocov_cos_WN(X_cos_WN)


#List of all MSEs
MSE_WN =[]
MSE_SUM_WN = []
MSE_GEO_SUM_WN = []
MSE_COS = []

#Mean square estimation
for T in [10,100,500,1000]:
    mse_wn_inter = []
    mse_sum_wn_inter = []
    mse_geo_wn_inter = []
    mse_cos_wn_inter = []
    for step in range(100):
        X_WN = white_nosie(number_var=T)
        X_sum_WN = sum_white_noise(number_variable=T)
        X_sum_geometric_WN = geometric_white_noise(number_variables=T)
        X_cos_WN = cos_noise(number_variables=T)

        theoretical_autocovariance_WN = theoretical_autocov_WN(X_WN)[0:T]
        theoretical_autocovariance_sum = theoretical_autocov_sum_WN(X_sum_WN)[0:T]
        theoretical_autocovariance_geo_sum = theoretical_autocov_geo_sum_WN(X_sum_geometric_WN)[0:T]
        theoretical_autocovariance_cos = theoretical_autocov_cos_WN(X_cos_WN)[0:T]

        empirical_autocovariance_WN = empirical_autocovariance(X_WN,taus=np.arange(T),mean=0)
        empirical_autocovariance_sum = empirical_autocovariance(X_sum_WN,mean=1, taus=np.arange(T))
        empirical_autocovariance_geo_sum = empirical_autocovariance(X_sum_geometric_WN,mean=1,taus=np.arange(T))
        empirical_autocovariance_cos = empirical_autocovariance(X_cos_WN,mean=0,taus=np.arange(T))

        mse_wn_inter.append(np.mean((theoretical_autocovariance_WN-empirical_autocovariance_WN)**2))
        mse_sum_wn_inter.append(np.mean((theoretical_autocovariance_sum-empirical_autocovariance_sum)**2))
        mse_geo_wn_inter.append(np.mean((theoretical_autocovariance_geo_sum-empirical_autocovariance_geo_sum)**2))
        mse_cos_wn_inter.append(np.mean((theoretical_autocovariance_cos-empirical_autocovariance_cos)**2))

    
    MSE_WN.append(mse_wn_inter)
    MSE_SUM_WN.append(mse_sum_wn_inter)
    MSE_GEO_SUM_WN.append(mse_geo_wn_inter)
    MSE_COS.append(mse_cos_wn_inter)

#TODO: For Cos , we have a HUGE gap BECAUSE ht e random variable is most likelt not weakly stationnary/ To prove !
# We have proven empirically that it is NOT wekaly stationnary

#plotting the empirical mean of various random variables 
#plt.grid()
#plt.plot(indexes,X_cos_WN, label='rv path', marker='H')
#plt.plot(indexes,np.full_like(indexes,empirical_mean_WN), label='empirical mean')
#plt.scatter(indexes,empirical_autocovariance_cos, label='empirical autocov', color='green' , marker = 'x')
#plt.scatter(indexes,theoretical_autocovariance_cos,label='theo autocov', color='red', marker='1')
#plt.legend()
#plt.show()

# Plotting the MSE for each T defined in the TP

LABELS = ['10','100','500','1000']
COLORS = ['peachpuff', 'orange', 'tomato','pink']
plt.grid()
#plt.boxplot(MSE_COS,patch_artist=True,tick_labels=LABELS)
#plt.boxplot(MSE_GEO_SUM_WN,patch_artist=True,tick_labels=LABELS)
ax = plt.boxplot(MSE_SUM_WN,patch_artist=True,tick_labels=LABELS)
#plt.boxplot(MSE_WN,patch_artist=True,tick_labels=LABELS)
for patch, color in zip(ax['boxes'], COLORS):
    patch.set_facecolor(color)
plt.show()