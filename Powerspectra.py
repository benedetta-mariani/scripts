import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15

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
os.chdir("../../../Downloads/xBenedetta/")
lista = []
for r in range(len(os.listdir())):
    if "mat" in os.listdir()[r]:
        lista.append(os.listdir()[r])
for dat in range(len(lista)):
    print('Evaluating ', lista[dat])
    mat = mat73.loadmat(lista[dat]);
    matr = mat['amplifier_data']
    stim = mat['adc_data']
    stim2 = mat['dig_in_data']
    t = np.arange(0,len(matr[0]))*0.00004
    dt = t[1]-t[0]
    fs = 1/dt
    nyq = 0.5*fs


    lowcut =  0.1
    highcut = 150
    low = lowcut / nyq
    high = highcut / nyq
    order = 2
    f2, h2 = signal.butter(order,[low, high], btype='bandpass')


    #newdata = np.empty((27,len(matr[0])))
    lfp = np.empty((27,len(matr[0])))

    for s in range(27):
        sig = matr[s,:]
        #newdata[s,:] = signal.filtfilt(f, h, sig, padlen=150)
        lfp[s,:] = signal.filtfilt(f2, h2, sig, padlen=150)


    indexes = (np.where(stim> 1.5)[0])[np.where(np.diff(np.where(stim> 1.5)[0])>1)]


    firstfreq= []
    secondfreq = []
    firstpowers = []
    secondpowers = []
    thirdfreq = []
    thirdpowers = []
    Pspect = []
    o = 0
    for l in range(len(indexes)):
        for el in range(27):
            x,p = signal.welch(lfp[el,indexes[l]  +  int(25000) :indexes[l]  +  int(25000*6)],fs = 1/0.00004, scaling = 'density', nperseg = int(25000) )
            #xmaxx = np.array(x)[np.array(findmaxima(Pxx)[:,0], dtype = int)][np.argsort(np.array(findmaxima(Pxx)[:,1]))[::-1]]
            idx = np.array(findmaxima(p)[:,0], dtype = int)[np.argsort(np.array(findmaxima(p)[:,1]))[::-1]]
            xmaxx = x[idx]
            pmax = p[idx]
            firstfreq.append(xmaxx[0])
            secondfreq.append(xmaxx[1])
            firstpowers.append(pmax[0])
            secondpowers.append(pmax[1])
            thirdfreq.append(xmaxx[2])
            thirdpowers.append(pmax[2])



            Pspect.append(p)
            if el == 18 and l == 10 and True:
                fig = plt.figure()
                #ax = fig.add_subplot(1,2,1)
                plt.plot(x,p, color ='darkblue')
                plt.xlim(0,30)
                plt.show()

    ps = np.mean(np.array(Pspect),0)
    xmaxx = np.array(x)[np.array(findmaxima(ps)[:,0], dtype = int)][np.argsort(np.array(findmaxima(ps)[:,1]))[::-1]]
    Pspect = np.array(Pspect)
    print(xmaxx[0],xmaxx[1])

    """
    plt.figure()
    plt.plot(x,ps, 'darkblue')
    plt.xlim(0,30)
    plt.fill_between(x,ps - np.std(Pspect,0)/np.sqrt(len(Pspect)),ps + np.std(Pspect,0)/np.sqrt(len(Pspect)),color = 'lightblue')
    plt.savefig('Power'+ label + 'stim.jpg', dpi = 300, bbox_inches = 'tight')
    plt.show()
    """

    """
    pspectrum = np.array(pspectrum)
    ps = np.mean(pspectrum,0)
    xmaxx = np.array(x)[np.array(findmaxima(ps)[:,0], dtype = int)][np.argsort(np.array(findmaxima(ps)[:,1]))[::-1]]

    print(xmaxx[0],xmaxx[1])
    plt.figure()
    plt.plot(x,ps,color = 'tab:blue', lw = 2)
    plt.fill_between(x,ps - np.std(pspectrum,0)/np.sqrt(len(pspectrum)),ps + np.std(pspectrum,0)/np.sqrt(len(pspectrum)),color = 'lightblue')

    #plt.plot(xmaxx, np.sort(np.array(findmaxima(ps)[:,1]))[::-1], 'ro')
    plt.xlim(0,40)
    plt.xlabel('Frequency')
    plt.ylabel('PSD')

    plt.savefig('Power'+ str(data) + 'rest.jpg', dpi = 300, bbox_inches = 'tight')
    #plt.show()

    #print(np.array(maxfreq).mean(), np.array(maxfreq2).mean(),  np.array(maxfreq3).mean())
    """
    Pspect = []
    firstfreqstim = []
    secondfreqstim = []
    firstpowersstim = []
    secondpowersstim = []
    thirdfreqstim = []
    thirdpowersstim = []
    o = 0
    maxfreq2stim = []
    for l in range(len(indexes)):
        for el in range(27):
            x,p = signal.welch(lfp[el,indexes[l] :indexes[l]  +  int(25000)],fs = 1/0.00004, scaling = 'density', nperseg = int(25000))
            idx = np.array(findmaxima(p)[:,0], dtype = int)[np.argsort(np.array(findmaxima(p)[:,1]))[::-1]]
            xmaxx = x[idx]
            pmax = p[idx]
            firstfreqstim.append(xmaxx[0])
            secondfreqstim.append(xmaxx[1])
            firstpowersstim.append(pmax[0])
            secondpowersstim.append(pmax[1])
            Pspect.append(p)
            thirdfreqstim.append(xmaxx[2])
            thirdpowersstim.append(pmax[2])
            if el == 18 and l == 10 and True:
                fig = plt.figure()
                #ax = fig.add_subplot(1,2,1)
                plt.plot(x,p, color ='darkblue')
                plt.xlim(0,30)
                plt.show()
                #ax.set_xlim(0,30)
            #ax = fig.add_subplot(1,2,2)
            #ax.plot(lfpTrials[r,s,:int(25000*0.5)])
            #plt.fill_between(x,ps - np.std(pspectrum,0)/np.sqrt(len(pspectrum)),ps + np.std(pspectrum,0)/np.sqrt(len(pspectrum)),color = 'lightblue')
    #plt.savefig('Power'+ label + 'stim.jpg', dpi = 300, bbox_inches = 'tight')
            #plt.show()
            #print(r,s)

    Pspect = np.array(Pspect)
    ps = np.mean(np.array(Pspect),0)
    xmaxx = np.array(x)[np.array(findmaxima(ps)[:,0], dtype = int)][np.argsort(np.array(findmaxima(ps)[:,1]))[::-1]]
    """
    print(xmaxx[0],xmaxx[1])
    plt.figure()
    plt.plot(x,ps, color=  'darkblue')
    plt.xlim(0,30)
    plt.fill_between(x,ps - np.std(Pspect,0)/np.sqrt(len(Pspect)),ps + np.std(Pspect,0)/np.sqrt(len(Pspect)),color = 'lightblue')
    plt.savefig('Power'+ label + 'Rest.jpg', dpi = 300, bbox_inches = 'tight')
    plt.show()
    """
    os.chdir("../../Desktop/Github/Oscillations analysis/Fig")

    import seaborn as sn
    import pandas as pd
    meanfirstpower = np.mean(firstpowers)
    meanfirstpowerstim = np.mean(firstpowersstim)

    data = pd.DataFrame(columns = ['Condition',  'Frequency'])
    data['Condition'] = [ f'Rest mp: {round(meanfirstpower,2)}' for r in range(len(firstfreq))] + [ f'Stim mp: {round(meanfirstpowerstim,2)}' for r in range(len(firstfreqstim))]
    data['Frequency'] = firstfreq + firstfreqstim

    plt.figure()
    sn.boxplot(x = "Condition", y = "Frequency", data = data, palette = 'Set2')
    plt.ylim(0,20)
    plt.savefig('First-frequencies' +str(dat) + '.jpg',bbox_inches = 'tight')

    meansecondpower = np.mean(secondpowers)
    meansecondpowerstim = np.mean(secondpowersstim)
    data = pd.DataFrame(columns = ['Condition',  'Frequency'])
    data['Condition'] = [ f'Rest mp: {round(meansecondpower,2)}' for r in range(len(secondfreq))] + [ f'Stim mp: {round(meansecondpowerstim,2)}' for r in range(len(secondfreqstim))]
    data['Frequency'] = secondfreq + secondfreqstim

    plt.figure()
    sn.boxplot(x = "Condition", y = "Frequency", data = data, palette = 'Set2')
    plt.ylim(0,20)
    plt.savefig('Second-frequencies' +str(dat) + '.jpg', bbox_inches = 'tight')


    meanthirdpower = np.mean(thirdpowers)
    meanthirdpowerstim = np.mean(thirdpowersstim)
    data = pd.DataFrame(columns = ['Condition',  'Frequency'])
    data['Condition'] = [ f'Rest mp: {round(meanthirdpower,2)}' for r in range(len(thirdfreq))] + [ f'Stim mp: {round(meanthirdpowerstim,2)}' for r in range(len(thirdfreqstim))]
    data['Frequency'] = thirdfreq + thirdfreqstim

    plt.figure()
    sn.boxplot(x = "Condition", y = "Frequency", data = data, palette = 'Set2')
    plt.ylim(0,20)
    plt.savefig('Third-frequencies' +str(dat) + '.jpg', bbox_inches = 'tight')

    os.chdir("../../../../Downloads/xBenedetta/")
    #plt.show()
    
