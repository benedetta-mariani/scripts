import os
os.chdir("../../../Downloads/xBenedetta/")
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

def my_moving_window(x, window=3, FUN=np.mean):
    """
    Calculates a moving estimate for a signal

    Args:
      x (numpy.ndarray): a vector array of size N
      window (int): size of the window, must be a positive integer
      FUN (function): the function to apply to the samples in the window
    Returns:
      (numpy.ndarray): a vector array of size N, containing the moving
      average of x, calculated with a window of size window
    """
    if len(x.shape) == 2:
        output = np.zeros(x.shape)
        for rown in range(x.shape[0]):
            output[rown, :] = my_moving_window(x[rown, :],window=window,FUN=FUN)                       
        return output
    output = np.zeros(x.size)
    for samp_i in range(x.size):
        values = []
        # loop through the window:
        for wind_i in range(int(1 - window), 1):
            if ((samp_i + wind_i) < 0) or (samp_i + wind_i) > (x.size - 1):
                # out of range
                continue
            # sample is in range and not nan, use it:
            if not(np.isnan(x[samp_i + wind_i])):
                values += [x[samp_i + wind_i]]
        # calculate the mean in the window for this point in the output:
        output[samp_i] = FUN(values)
    return output


def findmaxima(_input):
    maxima=[]
    for i in range(1, len(_input)-1):
        if (_input[i-1] < _input[i] and _input[i] > _input[i+1]): 
                maxima.append([i, _input[i]])
    maxima = np.array(maxima)
    return(maxima)

def R_helper_detector_LFP(_input):
    maxima=[]
    for i in range(1, len(_input)-1):
        if (_input[i-1] < _input[i] and _input[i] > _input[i+1]) and _input[i] > np.median(_input) : 
            maxima.append([i, _input[i]])
    maxima = np.array(maxima)
    return(maxima)


# Dataset 1 

import mat73
lista = []
for r in range(len(os.listdir())):
    if "mat" in os.listdir()[r]:
        lista.append(os.listdir()[r])

freq = float(input('Insert selected frequency'))
for data in range(len(lista)):
    print('Evaluating ', lista[data])
    mat = mat73.loadmat(lista[data]);
    matr = mat['amplifier_data']
    stim = mat['adc_data']
    stim2 = mat['dig_in_data']
    t = np.arange(0,len(matr[0]))*0.00004
    dt = t[1]-t[0]
    fs = 1/dt
    nyq = 0.5*fs
    lowcut =  freq-1 
    highcut = freq+1
    low = lowcut / nyq
    high = highcut / nyq
    order = 2
    f, h = signal.butter(order,[low, high], btype='bandpass')

    lowcut =  0.1
    highcut = 150
    low = lowcut / nyq
    high = highcut / nyq
    order = 2
    f2, h2 = signal.butter(order,[low, high], btype='bandpass')


    newdata = np.empty((27,len(matr[0])))
    #lfp = np.empty((27,len(matr[0])))

    for s in range(27):
        sig = matr[s,:]
        newdata[s,:] = signal.filtfilt(f, h, sig, padlen=150)
        #lfp[s,:] = signal.filtfilt(f2, h2, sig, padlen=150)


    indexes = (np.where(stim> 1.5)[0])[np.where(np.diff(np.where(stim> 1.5)[0])>1)]

    """
    maxfreq= []
    maxfreq2 = []
    maxfreq3 = []
    pspectrum = []
    o = 0
    for l in range(len(indexes)):
        for el in range(18,22):
            x,Pxx = signal.periodogram(lfp[el,indexes[l]  +  int(25000*6) :indexes[l]  +  int(25000*9)],fs = 1/0.00004)
            xmaxx = np.array(x)[np.array(findmaxima(Pxx)[:,0], dtype = int)][np.argsort(np.array(findmaxima(Pxx)[:,1]))[::-1]]
            maxfreq.append(xmaxx[0])
            maxfreq2.append(xmaxx[1])
            maxfreq3.append(xmaxx[2])
            pspectrum.append(Pxx)
            
            
            o += 1
            if o < 6:
                plt.figure()
                xmaxx = np.array(x)[np.array(findmaxima(Pxx)[:,0], dtype = int)]
                maxx = np.array(findmaxima(Pxx)[:,1])
                plt.plot(xmaxx, maxx, 'ro')
                plt.plot(x,Pxx)
                plt.xlim(0,20)
            
    pspectrum = np.array(pspectrum)
    ps = np.mean(pspectrum,0)
    #plt.figure()
    #plt.plot(x,ps,'r', lw = 2)
    #plt.xlim(0,30)
    #plt.show()

    print(np.array(maxfreq).mean(), np.array(maxfreq2).mean(),  np.array(maxfreq3).mean())

    pspectrum = []
    maxfreqstim = []
    o = 0
    maxfreq2stim = []
    for l in range(len(indexes)):
        for el in range(18,22):
            x,Pxx = signal.periodogram(lfp[el,indexes[l] :indexes[l]  +  int(25000*0.5)],fs = 1/0.00004)
            xmaxx = np.array(x)[np.array(findmaxima(Pxx)[:,0], dtype = int)][np.argsort(np.array(findmaxima(Pxx)[:,1]))[::-1]]
            maxfreqstim.append(xmaxx[0])
            maxfreq2stim.append(xmaxx[1])
            pspectrum.append(Pxx)
            
            o += 1
            if o < 6:
                plt.figure()
                xmaxx = np.array(x)[np.array(findmaxima(Pxx)[:,0], dtype = int)]
                maxx = np.array(findmaxima(Pxx)[:,1])
                plt.plot(xmaxx, maxx, 'ro')
                plt.plot(x,Pxx)
                plt.xlim(0,20)
            
    #pspectrum = np.array(pspectrum)
    #ps = np.mean(pspectrum,0)
    #plt.figure()
    #plt.plot(x,ps,'r', lw = 2)
    #plt.xlim(0,30)
    #plt.show()
    print(np.array(maxfreqstim).mean(), np.array(maxfreq2stim).mean())
    """
    ## Phase response curve
    #num = int(indexes[0]/25000)
    trials = []
    t_s = 25000*7
    for r in range(1,len(indexes)):
        for el in range(27):
            trials.append(newdata[el,indexes[r]-t_s:indexes[r]+25000*4])
    trials = np.array(trials)
    print(trials.shape)
    phasestim = []
    phases = []
    
    #t_s = 3051
    t = []

    for trial in range(1,trials.shape[0]):
        #for elec in range(27):
        
        peaks =R_helper_detector_LFP(trials[trial,:])
        #print(peaks.shape)
        """
        if trial < 6:
            plt.figure()
            plt.plot(trials[trial], 'orange', alpha = 0.3)

            plt.plot(peaks[:,0],peaks[:,1], 'ro')
            plt.vlines(t_s,-600,600, 'r', lw = 2)
            #plt.xlim(1000,1500)
        """
        idx = np.where( (t_s- peaks[:,0]) == (t_s- peaks[:,0])[(t_s- peaks[:,0])>0].min())[0][0]
        time = peaks[:,0][idx]
        ts = t_s - time
        #t_s = 1110
        T0 =[]
        for r in range(len(peaks[:idx])):
            #print(len(peaks[:idx-1]))
            T0.append(peaks[:,0][idx-r] - peaks[:,0][idx -1-r])
        T0 = np.array(T0).mean()
        t.append(T0)

        T1 = peaks[:,0][idx + 1] - peaks[:,0][idx]
        phase = (T0 - T1)/T0*2*np.pi
        phase_stim = ts/T0*2*np.pi
        #if phase_stim > 2*np.pi: o+= 1
        f1 = np.sign(phase_stim)*(np.abs(phase_stim))%(2*np.pi) #?
        #print(phase_stim)
        #print(np.angle(np.exp(1j*phase_stim)) + np.pi, np.sign(phase_stim)*(np.abs(phase_stim))%(2*np.pi))
        phasestim.append(f1)
        #f2 = np.sign(phase)*(np.abs(phase))
        phases.append(np.angle(np.exp(1j*phase)))
        """
        if f2 > np.pi:
            phases.append(f2 - 2*np.pi)
        elif f2 < -np.pi:
            phases.append(f2 + 2*np.pi)
        else:
            phases.append(f2)
        """
    print(1/(np.array(t).mean()*0.00004))
    phases = np.array(phases)[~np.isnan(np.array(phases))]
    phasestim = np.array(phasestim)[~np.isnan(np.array(phasestim))]
    phases = phases[np.argsort(phasestim)]
    phasestim = np.sort(phasestim)

    for g in range(len(phasestim)):
        phasestim[g] = round(phasestim[g],3)
    uni = np.unique(phasestim)
    uniphases = [[] for r in range(len(uni))]

    for l in range(len(uni)):
        for r in range(len(phases)):
            if phasestim[r] == uni[l]:
                uniphases[l].append(phases[r])

    means = []
    stds =[]
    for r in range(len(uniphases)):
        means.append(np.mean(uniphases[r]))
        stds.append(np.std(uniphases[r]))    

    means,stds = np.array(means), np.array(stds)
    #uniphases = np.array(uniphases)

    mean = my_moving_window(means,10)
    std = my_moving_window(stds,10)
    unii = my_moving_window(uni,10)

    ## Fit of Fourier Coefficients

    y = np.array(means)
    oss01 = np.ones((len(uni)))
    oss1 = np.cos(uni)
    oss2 = np.cos(2*uni)
    #oss02 = np.ones((len(unii)))
    oss3 = np.sin(uni)
    oss4 = np.sin(2*uni)

    X = np.vstack((oss1,oss2,oss3,oss4,oss01)).T
    a1,a2,b1,b2,a0= (np.linalg.inv(X.T @ X) @ X.T) @ y
    
    plt.figure()
    #plt.fill_between(unii, mean - 2*std, mean + 2*std, color = 'lightblue')
    plt.scatter(unii,mean,c = 'lightblue')

    theta = np.arange(0,2*np.pi,0.1)
    plt.plot(theta, a0 + a1*np.cos(theta) + a2*np.cos(2*theta)+  b1*np.sin(theta) + b2*np.sin(2*theta), 'k', lw = 2 )
    #plt.plot(theta, a01 + a11*np.cos(theta) + a21*np.cos(2*theta)+  b11*np.sin(theta) + b21*np.sin(2*theta),color = 'orange' )

    plt.xticks(np.arange(0,2*np.pi + np.pi/2,np.pi/2), labels = ["0", "$\pi/2$","$\pi$","$3/2\pi$","$2\pi$"])
    plt.yticks(np.arange(-np.pi,np.pi + np.pi/2,np.pi/2), labels = ["$-\pi$","$\pi/2$","$0$","$\pi/2$","$\pi$"])
    plt.xlabel('Phase at stimulation time')
    plt.ylabel(r'$\Delta \phi$');
    
    try:
        os.chdir('../../Desktop/Github/Oscillations analysis/Fig')
    except:
        os.mkdir('../../Desktop/Github/Oscillations analysis/Fig')
        os.chdir('../../Desktop/Github/Oscillations analysis/Fig')
    plt.savefig('PRCurethane' + str(freq) +'Hz' + str(data) +'.jpg', dpi = 300, bbox_inches ='tight')
    #plt.show()

    os.chdir('../../../../Downloads/xBenedetta')
    del mat,matr,newdata

