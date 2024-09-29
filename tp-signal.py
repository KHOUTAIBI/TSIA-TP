import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy, time
import scipy.signal
from scipy.io.wavfile import write
import os
import pyaudio
import wave, struct

def load_sound(file):
    return wave.open(file, 'rb')


def play_sound(file, chunk = 1024):
    """
    Script from PyAudio doc
    """
    wf = load_sound(file)
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(chunk)

    while data:
        stream.write(data)
        data = wf.readframes(chunk )

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    
def plot_sound(data, times, name='default_name', save=False):
    plt.figure(figsize=(30, 4))
    plt.fill_between(times, data)
    plt.xlim(times[0], times[-1])
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    if save:
        plt.savefig(name+'.png', dpi=100)
    plt.show()

data_path = os.getcwd()
filename = 'caravan_48khz.wav'
sound = os.path.join(data_path, filename) 



wavefile = load_sound(sound)
print(wavefile.getparams())


Fs = int(wavefile.getframerate())
num_samples = int(wavefile.getnframes())
data = wavefile.readframes(num_samples)
data = struct.unpack('{n}h'.format(n=num_samples), data)
x = np.array(data)


def oversample(X,step):
    oversampled = np.zeros(len(X) * step)
    oversampled[::step] = X  # Place elements of X at every other position
    return oversampled


def downsampling(X,step):
    return X[::step]
        

def filter(data):
    X_over = oversample(data,2)
    ranges = np.arange(len(data))
    sinc_filter = (2/6)*np.sinc(2*np.pi*1/6*ranges)
    X_conv = scipy.signal.fftconvolve(X_over,sinc_filter,mode='same')
    print("finished")
    X_down = downsampling(X_conv,3)
    return X_down


#X = filter(data)
#
#write('new_caravan.wav', Fs*2//3, np.array(X, dtype=np.int16)) # to write a new wave file
#
#print("finished uploading music")
#
#timestep = 1/float(Fs*2//3)
#times = np.arange(len(X))*timestep



#plot_sound(X,times)


def unoptimized_shift(X,step1,step2):

    start = time.time()

    E_0 = np.array([1 if i%2 == 0 else 0 for i in range(len(X))])
    E_1 = np.array([1 if i%2 == 1 else 0 for i in range(len(X))])
    E_1 = np.roll(E_1,1)
    
    X_over = oversample(X, step1)

    X_result = scipy.signal.lfilter(E_0,[1],X_over) + scipy.signal.lfilter(E_1,[1],X_over)

    X_result = downsampling(X_result,step2)

    end = time.time()

    print(f"Execution time : {end-start}")

    return X_result


def optimized_shift(X,step1, step2):

    start = time.time()

    E_0 = np.array([1 if i%2 == 0 else 0 for i in range(len(X))])
    E_1 = np.array([1 if i%2 == 1 else 0 for i in range(len(X))])

    X = X.astype(float)
    
    X_1 = scipy.signal.lfilter(E_0,[1],X)
    X_2 = scipy.signal.lfilter(E_1,[1],X)
    
    X_1 = oversample(X_1, step=step1)
    X_2 = oversample(X_2, step=step1)

    X_2 = np.roll(X_2, 1)
    
    X_out = X_1 + X_2
    
    X_out = downsampling(X_out, step=step2)

    end = time.time()

    print(f"Exectuion time: {end-start}")

    return X_out


#compute the outputs and compare the execution time, may take up to 20 seconds !
#X = unoptimized_shift(X=x[:],step1=2,step2=3)
#X_out = optimized_shift(X=x[:], step1=2, step2=3)


##Defining the variables that we will use in the STFT

N = x.shape[0] # % longueur du signal
Nw = 512
w = np.hanning(512) # définition de la fenetre d'analyse
ws = w.copy; # définition de la fenêtre de synthèse
R = 1 # incrément sur les temps d'analyse, appelé hop size, t_a=uR
M = 32 # ordre de la tfd
L = M/2+1
affich = 1 ; # pour affichage du spectrogramme, 0 pour
             # pour faire analyse/modif/synthèse sans affichage
             # note: cf. spectrogram sous Matlab
Nt = np.rint((N - Nw) / R) # calcul du nombre de tfd à calculer
Nt = Nt.astype(int)
y = np.zeros((N,1)) # signal de synthèse

TFW = np.fft.fft(w,M)
TFW = np.abs(np.fft.fftshift(TFW))
with np.errstate(divide='ignore', invalid='ignore'):
    TFW = 20 * np.log10(TFW)

indexes = np.arange(len(TFW))

#plt.grid()
##plt.plot(indexes,w)
#plt.plot(indexes,TFW)
#plt.show()


if affich:
    Xtilde = np.zeros((M,Nt),dtype=complex)


for u in np.arange(0,Nt).reshape(-1): # boucle sur les trames
    deb = u * R + 1 # début de trame
    fin = deb + Nw # fin de trame
    tx = np.multiply(x[np.arange(deb.astype(int),fin.astype(int))],w) # calcul de la trame 
    X = np.fft.fft(tx,M) # tfd à l'instant b
    if affich:
        Xtilde[:,u] = X
    # opérations de transformation (sur la partie \nu > 0)
    # ....
    
    h = [M, [0, -1/6, 1/6, 0.5] , ]

    Y = X.copy
    # fin des opérations de transformation
    # resynthèse
    # overlap add

def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

#f, t, Sxx = scipy.signal.spectrogram(np.abs(Xtilde[3,]), Fs)
#
#plt.pcolormesh(t, f, Sxx, shading='gouraud')
#
#plt.ylabel('Frequency [Hz]')
#
#plt.xlabel('Time [sec]')
#
#plt.show()


write('STFT_noise.wav', Fs, np.array(np.real(Xtilde[3,]), dtype=np.int16)) # to write a new wave file

print(Xtilde[3,])

def ola(w = None,hop = None,Nb = 10): 
# function output = ola(w,hop,Nb)
# realise l'addition-recouvrement de la fenetre w,
# avec un décalage hop et un nombre Nb de fenetres.
# par defaut Nb = 10;
    
    w = w[:, np.newaxis]
    N = len(w)
    output = np.zeros(((Nb - 1) * hop + N,1)) # réserve l'espace memoire
    
    for k in np.arange(0,Nb).reshape(-1):
        deb = k* hop
        fin = deb + N
        output[np.arange(deb,fin)] += + w # OLA
    
    return output

w = np.hanning(512)

#verify the tps sufficient condition
for u in range(len(Xtilde[:,0])): 
    reconstruciton = np.fft.ifft(Xtilde[:,u])

print(reconstruciton)


reconstrcuted_signal = ola(w=reconstruciton,hop=512,Nb=len(Xtilde[:,0]))
indexes = np.arange(len(reconstruciton))

plt.grid()
fig , ax = plt.subplots(2)
ax[0].plot(indexes,reconstruciton)
ax[1].plot(np.arange(len(x)),x)
plt.show()


