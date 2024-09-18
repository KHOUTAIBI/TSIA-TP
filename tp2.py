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
def cos_noise(n,A0,lambda0):
    T=np.arange(0,n) # An array of the number of random variable
    return A0*np.cos(lambda0*T+np.random.uniform(0,2*np.pi,n))+np.random.normal(0,1,n) # returns the cos function


##We compute I defined in the Tps questions
def I(X,m,k):
    TFX = np.fft.fft(X,m) #compute the FFT of the variable X
    scaling_factor = 1 / (2 * np.pi * len(X)) # scale the factors 
    return scaling_factor  * np.absolute(TFX)**2 #return the arrays of the module of the FFT


X_sum_white_noise = sum_white_noise(1,1,100)
X_sum_geometric = geometric_white_noise(K=50,a=1,number_variables=100)
X_cos = cos_noise(n=100,A0=1,lambda0=0.6)
X = white_nosie(0,1,100)
Tf = I(X_sum_white_noise,None,100)
indexes = np.arange(0,100)


plt.grid()
plt.plot(indexes,Tf,marker='H',color= 'pink')
plt.show()

