#%%
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
#from getdist import plots, MCSamples
import scipy
from scipy.optimize import differential_evolution
import numpy as np
import xarray as xr
from getdist import plots, MCSamples
import time
from copy import deepcopy
import multiprocessing as mp
import pandas as pd
import os
import h5py
import sys
sys.path.append('/cluster/home/sstrub/Repositories/LDC/lib/lib64/python3.8/site-packages/ldc-0.1-py3.8-linux-x86_64.egg')

from ldc.lisa.noise import get_noise_model
from ldc.common.series import TimeSeries, window
import ldc.waveform.fastGB as fastGB
# from ldc.common.tools import compute_tdi_snr

from fastkde import fastKDE
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from chainconsumer import ChainConsumer

# customized settings
plot_parameter = {  # 'backend': 'ps',
    "font.family": "serif",
    "font.serif": "times",
    "font.size": 16,
    "axes.labelsize": "medium",
    "axes.titlesize": "medium",
    "legend.fontsize": "medium",
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "grid.color": "k",
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "savefig.dpi": 150,
}

# tell matplotlib about your param_plots
rcParams.update(plot_parameter)
# set nice figure sizes
fig_width_pt = 1.5*464.0  # Get this from LaTeX using \showthe\columnwidth
golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio
ratio = golden_mean
inches_per_pt = 1.0 / 72.27  # Convert pt to inches
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * ratio  # height in inches
fig_size = [fig_width, fig_height]
fig_size_squared = [fig_width, fig_width]
rcParams.update({"figure.figsize": fig_size})

def Window(tm, offs=1000.0):
    xl = offs
    ind_r = np.argwhere(tm[-1] - tm <= offs)[0][0]
    xr = tm[ind_r]
    kap = 0.005
    winl = 0.5 * (1.0 + np.tanh(kap * (tm - xl)))
    winr = 0.5 * (1.0 - np.tanh(kap * (tm - xr)))
    return winl * winr

def scaletooriginal(previous_max, boundaries):
    maxpGB = {}
    for parameter in parameters:
        if parameter in ["EclipticLatitude"]:
            maxpGB[parameter] = np.arcsin((previous_max[parameters.index(parameter)] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0])
        elif parameter in ["Inclination"]:
            value = (previous_max[parameters.index(parameter)] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0]
            if value > 1:
                value -= 2
            elif value < -1:
                value += 2
            maxpGB[parameter] = np.arccos(value)
        elif parameter in parameters_log_uniform:
            maxpGB[parameter] = 10**((previous_max[parameters.index(parameter)] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0])
        else:
            maxpGB[parameter] = (previous_max[parameters.index(parameter)] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0]
    return maxpGB

def scaletooriginalparameter(previous_max, boundaries):
    maxpGB = {}
    for parameter in parameters:
        if parameter in ["EclipticLatitude"]:
            maxpGB[parameter] = np.arcsin((previous_max[parameter] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0])
        elif parameter in ["Inclination"]:
            maxpGB[parameter] = np.arccos((previous_max[parameter] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0])
        elif parameter in parameters_log_uniform:
            maxpGB[parameter] = 10**((previous_max[parameter] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0])
        else:
            maxpGB[parameter] = (previous_max[parameter] * (boundaries[parameter][1] - boundaries[parameter][0])) + boundaries[parameter][0]
    return maxpGB

def scaleto01(previous_max, boundaries):
    maxpGB = {}
    for parameter in parameters:
        if parameter in ["EclipticLatitude"]:
            maxpGB[parameter] = (np.sin(previous_max[parameter]) - boundaries[parameter][0]) / (boundaries[parameter][1] - boundaries[parameter][0])
        elif parameter in ["Inclination"]:
            maxpGB[parameter] = (np.cos(previous_max[parameter]) - boundaries[parameter][0]) / (boundaries[parameter][1] - boundaries[parameter][0])
        elif parameter in parameters_log_uniform:
            maxpGB[parameter] = (np.log10(previous_max[parameter]) - boundaries[parameter][0]) / (boundaries[parameter][1] - boundaries[parameter][0])
        else:
            maxpGB[parameter] = (previous_max[parameter] - boundaries[parameter][0]) / (boundaries[parameter][1] - boundaries[parameter][0])
    return maxpGB

def Reduce_boundaries(maxpGB, boundaries, ratio=0.1):
    boundaries_reduced = deepcopy(boundaries)
    for parameter in parameters:
        length = boundaries[parameter][1] - boundaries[parameter][0]
        if parameter == "EclipticLatitude":
            boundaries_reduced[parameter] = [
                np.sin(maxpGB[parameter]) - length * ratio / 2,
                np.sin(maxpGB[parameter]) + length * ratio / 2,
            ]
        elif parameter == "Inclination":
            boundaries_reduced[parameter] = [
                np.cos(maxpGB[parameter]) - length * ratio / 2,
                np.cos(maxpGB[parameter]) + length * ratio / 2,
            ]
        elif parameter == "Frequency":
            boundaries_reduced[parameter] = [
                maxpGB[parameter] - length * ratio / 2,
                maxpGB[parameter] + length * ratio / 2,
            ]
        elif parameter == "FrequencyDerivative":
            # boundaries_reduced[parameter] = [boundaries[parameter][0],-16.5]
            boundaries_reduced[parameter] = [
                maxpGB[parameter] - length * ratio / 2,
                maxpGB[parameter] + length * ratio / 2,
            ]
        elif parameter == "Amplitude":
            boundaries_reduced[parameter] = [
                np.log10(maxpGB[parameter]) - length * ratio / 2,
                np.log10(maxpGB[parameter]) + length * ratio / 2,
            ]
        elif parameter in ["InitialPhase",'Polarization']:
            boundaries_reduced[parameter] = [
                maxpGB[parameter] - length * ratio / 2,
                maxpGB[parameter] + length * ratio / 2,
            ]
        elif parameter == "EclipticLongitude":
            boundaries_reduced[parameter] = [
                maxpGB[parameter] - length * ratio / 2,
                maxpGB[parameter] + length * ratio / 2,
            ]
        if boundaries_reduced[parameter][0] < boundaries[parameter][0]:
            boundaries_reduced[parameter][0] = boundaries[parameter][0]
        if boundaries_reduced[parameter][1] > boundaries[parameter][1]:
            boundaries_reduced[parameter][1] = boundaries[parameter][1]
    return boundaries_reduced

def CoordinateMC(n, pGBs, boundaries, parameters_recorded, loglikelihood, n_trials=50):
    maxpGB = []
    parameters_recorded1 = []
    no_improvement_counter = 0
    for i in range(number_of_signals):
        maxpGB.append(deepcopy(pGBs))
        parameters_recorded1.append({})
        for parameter in parameters_no_amplitude:
            parametervalue = np.random.uniform(boundaries[parameter][0], boundaries[parameter][1])
            maxpGB[i][parameter] = parametervalue
            if parameter in ["EclipticLatitude"]:
                maxpGB[i][parameter] = np.arcsin(parametervalue)
            elif parameter in ["Inclination"]:
                maxpGB[i][parameter] = np.arccos(parametervalue)
            elif parameter in parameters_log_uniform:
                maxpGB[i][parameter] = 10**(parametervalue)
            parameters_recorded1[i][parameter] = []
            parameters_recorded1[i][parameter].append(maxpGB[i][parameter])
        parameters_recorded1[i]["Loglikelihood"] = []
    # if n == 0:
    #     maxpGB = deepcopy(pGBs)
    try:
        best_value
    except:
        best_value = loglikelihood(maxpGB)
    previous_best = deepcopy(best_value)
    maxpGB2 = deepcopy(maxpGB)
    for i in range(100):
        # if i > 48:
        #     n_trials = 50
        parameter1 = parameters_no_amplitude[i % 7]
        parameter2 = parameters_no_amplitude[np.random.randint(0, 6)]
        parameter3 = parameters_no_amplitude[np.random.randint(0, 6)]
        # parameter2 = 'InitialPhase'
        # parameter1 = 'Inclination'
        while parameter2 == parameter1:
            parameter2 = parameters_no_amplitude[np.random.randint(0, 6)]
        while parameter3 == parameter1 or parameter3 == parameter2:
            parameter3 = parameters_no_amplitude[np.random.randint(0, 6)]
        # if parameter1 == 'Frequency':
        #     parameter2 = 'Polarization'
        changeableparameters = [parameter1, parameter2]
        signal = i % number_of_signals
        for j in range(n_trials):
            k = 0
            for parameter in changeableparameters:
                parametervalue = np.random.uniform(boundaries[parameter][0], boundaries[parameter][1],1)[0]
                # if k == 0:
                #     parametervalue = np.random.uniform(j%number_per_row * 1/number_per_row, (j%number_per_row+1)*1/number_per_row)
                # else:
                #     parametervalue = np.random.uniform(int(j/number_per_row) * 1/number_per_row, (int(j/number_per_row)+1)*1/number_per_row)
                # parametervalue = parametervalue * (boundaries[parameter][1] - boundaries[parameter][0]) + boundaries[parameter][0]
                maxpGB2[signal][parameter] = parametervalue
                if parameter in ["EclipticLatitude"]:
                    maxpGB2[signal][parameter] = np.arcsin(parametervalue)
                elif parameter in ["Inclination"]:
                    maxpGB2[signal][parameter] = np.arccos(parametervalue)
                elif parameter in parameters_log_uniform:
                    maxpGB2[signal][parameter] = 10**(parametervalue)
                k += 1
            suggestion = loglikelihood(maxpGB2)
            if suggestion > previous_best:
                no_improvement_counter = 0
                previous_best = suggestion
                for parameter in changeableparameters:
                    maxpGB[signal][parameter] = maxpGB2[signal][parameter]
        if no_improvement_counter != 0:
            no_improvement_counter += 1
        if no_improvement_counter > 16:
            past_mean = 0
            sum_count = 0
            for l in range(n):
                try:
                    past_mean += parameters_recorded[l]["Loglikelihood"][i]
                    sum_count += 1
                except:
                    pass
            try:
                past_mean = past_mean / sum_count
                if previous_best > past_mean:
                    no_improvement_counter = 0
                else:
                    # print("no improvement")
                    break
            except:
                # no_improvement_counter = 0
                break

        if i in [30,40,50,60,70]:
            past_mean = 0
            sum_count = 0
            for l in range(n):
                try:
                    past_mean += parameters_recorded[l][0]["Loglikelihood"][i]
                    sum_count += 1
                except:
                    pass
            try:
                past_mean = past_mean / sum_count
                if previous_best > past_mean:
                    pass
                else:
                    break
            except:
                pass

        # start = time.time()
        # print(n, i, previous_best, loglikelihood(maxpGB), maxpGB)
        parameters_recorded1[0]["Loglikelihood"].append(loglikelihood(maxpGB))
        for i in range(number_of_signals):
            for parameter in parameters_no_amplitude:
                parameters_recorded1[i][parameter].append(maxpGB[i][parameter])

        maxpGB2 = deepcopy(maxpGB)
    if previous_best > best_value:
        best_value = previous_best
    parameters_recorded[n] = parameters_recorded1
    return parameters_recorded1

class Distribution(object):
    """
    draws samples from a one dimensional probability distribution,
    by means of inversion of a discrete inverstion of a cumulative density function

    the pdf can be sorted first to prevent numerical error in the cumulative sum
    this is set as default; for big density functions with high contrast,
    it is absolutely necessary, and for small density functions,
    the overhead is minimal

    a call to this distibution object returns indices into density array
    """
    def __init__(self, pdf, sort = True, interpolation = True, transform = lambda x: x):
        self.shape          = pdf.shape
        self.pdf            = pdf.ravel()
        self.sort           = sort
        self.interpolation  = interpolation
        self.transform      = transform

        #a pdf can not be negative
        assert(np.all(pdf>=0))

        #sort the pdf by magnitude
        if self.sort:
            self.sortindex = np.argsort(self.pdf, axis=None)
            self.pdf = self.pdf[self.sortindex]
        #construct the cumulative distribution function
        self.cdf = np.cumsum(self.pdf)
    @property
    def ndim(self):
        return len(self.shape)
    @property
    def sum(self):
        """cached sum of all pdf values; the pdf need not sum to one, and is imlpicitly normalized"""
        return self.cdf[-1]
    def __call__(self, N):
        """draw """
        #pick numbers which are uniformly random over the cumulative distribution function
        choice = np.random.uniform(high = self.sum, size = N)
        #find the indices corresponding to this point on the CDF
        index = np.searchsorted(self.cdf, choice)
        pdfs = self.pdf[index]
        #if necessary, map the indices back to their original ordering
        if self.sort:
            index = self.sortindex[index]
        #map back to multi-dimensional indexing
        index = np.unravel_index(index, self.shape)
        index = np.vstack(index)
        #is this a discrete or piecewise continuous distribution?
        if self.interpolation:
            index = index + np.random.uniform(size=index.shape)
        return self.transform(index), pdfs

class Search():
    def __init__(self,tdi_fs,Tobs, lower_frequency, upper_frequency, recombination=0.75):
        self.tdi_fs = tdi_fs
        self.GB = fastGB.FastGB(delta_t=dt, T=Tobs)
        self.reduced_frequency_boundaries = None
        self.recombination = recombination
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.padding = (upper_frequency - lower_frequency)/2
        tdi_ts = xr.Dataset(dict([(k, tdi_fs[k].ts.ifft()) for k in ["X", "Y", "Z"]]))
        # f, psdX =  scipy.signal.welch(tdi_ts["X"], fs=1.0/dt, window='hanning', nperseg=len(tdi_ts["X"])/1)
        # f, psdY =  scipy.signal.welch(tdi_ts["Y"], fs=1.0/dt, window='hanning', nperseg=len(tdi_ts["X"])/1)
        # f, psdZ =  scipy.signal.welch(tdi_ts["Z"], fs=1.0/dt, window='hanning', nperseg=len(tdi_ts["X"])/1)
        # psd = psdX + psdY + psdZ
        # indexes = np.logical_and(f>lower_frequency-self.padding, f<upper_frequency+self.padding)
        # psd = psd[indexes]
        self.frequency_T_threshold = 19.1*10**-3/2
        
        self.use_T_component = False
        if self.upper_frequency + self.padding > self.frequency_T_threshold:
            self.use_T_component = True
        # plt.figure()
        # plt.plot(f[indexes], psdX[indexes])
        # plt.plot(f[indexes], psd)
        # plt.show()

        frequencyrange =  [lower_frequency-self.padding, upper_frequency+self.padding]
        indexes = np.logical_and(tdi_fs['X'].f > frequencyrange[0], tdi_fs['X'].f < frequencyrange[1]) 
        self.dataX = tdi_fs["X"][indexes]
        self.dataY = tdi_fs["Y"][indexes]
        self.dataZ = tdi_fs["Z"][indexes]

        self.DAf = (self.dataZ - self.dataX)/np.sqrt(2.0)
        self.DEf = (self.dataZ - 2.0*self.dataY + self.dataX)/np.sqrt(6.0)
        self.DTf = (self.dataZ + self.dataY + self.dataX)/np.sqrt(3.0)

        # plt.figure()
        # plt.plot(f[indexes], np.abs(self.DAf))
        # plt.plot(f[indexes], np.abs(self.DEf))
        # plt.plot(f[indexes], np.abs(self.DAf)+np.abs(self.DEf))
        # plt.show()
        # print('frequencyrange',frequencyrange)
        fmin, fmax = float(self.dataX.f[0]), float(self.dataX.f[-1] + self.dataX.attrs["df"])
        freq = np.array(self.dataX.sel(f=slice(fmin, fmax)).f)
        Nmodel = get_noise_model(noise_model, freq)
        self.Sn = Nmodel.psd(freq=freq, option="X")
        self.SA = Nmodel.psd(freq=freq, option="A")
        self.SE = Nmodel.psd(freq=freq, option="E")
        self.ST = Nmodel.psd(freq=freq, option="T")

        f_0 = fmin
        f_transfer = 19.1*10**-3
        snr = 7
        amplitude_lower = 2*snr/(Tobs * np.sin(f_0/ f_transfer)**2/self.SA[0])**0.5
        snr = 1000
        amplitude_upper = 2*snr/(Tobs * np.sin(f_0/ f_transfer)**2/self.SA[0])**0.5
        amplitude = [amplitude_lower, amplitude_upper]
        # print('lower frequency', lower_frequency)
        # print('amplitude boundaries', amplitude)
        # print('amplitude boundaries previous', np.sqrt(np.max(psd))/1000,np.sqrt(np.max(psd))/10)
        # print('amplitude boundaries previous', np.max(np.abs(self.DAf.data))/10**7,np.max(np.abs(self.DAf.data))/10**5)
        # amplitude = np.sqrt(np.max(psd))

        # indexes = np.argsort(p.get('Frequency'))
        # index_low = np.searchsorted(p.get('Frequency')[indexes], frequencyrange[0])
        # index_high = np.searchsorted(p.get('Frequency')[indexes], frequencyrange[1])

        # pGB = deepcopy(pGBadded)
        # self.pGB = {'Amplitude': 3.676495e-22, 'EclipticLatitude': 0.018181, 'EclipticLongitude': 1.268061, 'Frequency': 0.01158392, 'FrequencyDerivative': 8.009579e-15, 'Inclination': 0.686485, 'InitialPhase': 4.201455, 'Polarization': 2.288223}
        
        fd_range = [np.log10(frequency_derivative(lower_frequency,0.1)),np.log10(frequency_derivative(lower_frequency,M_chirp_upper_boundary))]
        # fd_range = [frequency_derivative(lower_frequency,0.1),frequency_derivative(lower_frequency,M_chirp_upper_boundary)]
        # fd_range = [frequency_derivative_tyson_lower(lower_frequency),frequency_derivative_tyson(lower_frequency)]
        fd_range = [frequency_derivative_tyson_lower(lower_frequency),frequency_derivative(upper_frequency,M_chirp_upper_boundary)]
        self.boundaries = {
            "Amplitude": [np.log10(amplitude[0]),np.log10(amplitude[1])],
            # "Amplitude": [-23.5,-21],
            # "Amplitude": [np.log10(self.pGB['Amplitude'])-2,np.log10(self.pGB['Amplitude'])+1],
            "EclipticLatitude": [-1.0, 1.0],
            "EclipticLongitude": [-np.pi, np.pi],
            # "Frequency": [self.pGB["Frequency"] * 0.99995, self.pGB["Frequency"] * 1.00015],
            # "Frequency": [self.pGB["Frequency"] - 3e-7, self.pGB["Frequency"] + 3e-7],
            "Frequency": frequencyrange,
            "FrequencyDerivative": fd_range,
            # "FrequencyDerivative": [np.log10(5e-6*self.pGB['Frequency']**(13/3)),np.log10(8e-8*self.pGB['Frequency']**(11/3))],
            "Inclination": [-1.0, 1.0],
            "InitialPhase": [0.0, 2.0 * np.pi],
            "Polarization": [0.0, 1.0 * np.pi],
        }
        # print('fd_rage',self.boundaries['FrequencyDerivative'], 'at f=', lower_frequency)

        # print('fd * Tobs', 10**self.boundaries['FrequencyDerivative'][1] * Tobs)
        # print('smear f', 300*lower_frequency * 10**3 / 10**9)

        if self.boundaries['FrequencyDerivative'][0] > self.boundaries['FrequencyDerivative'][1]:
            c = self.boundaries['FrequencyDerivative'][0]
            self.boundaries['FrequencyDerivative'][0] = self.boundaries['FrequencyDerivative'][1]
            self.boundaries['FrequencyDerivative'][1] = c
        
        previous_max = np.random.rand(8)
        # previous_max[0] = np.random.rand(1)*0.1 +0.5
        # previous_max[3] = np.random.rand(1)*0.1 +0.5
        i = 0
        self.pGBs = {}
        for parameter in parameters:
            # if parameter in ["FrequencyDerivative"]:
            #     i -= 1
            if parameter in ["EclipticLatitude"]:
                self.pGBs[parameter] = np.arcsin((previous_max[i] * (self.boundaries[parameter][1] - self.boundaries[parameter][0])) + self.boundaries[parameter][0])
            elif parameter in ["Inclination"]:
                self.pGBs[parameter] = np.arccos((previous_max[i] * (self.boundaries[parameter][1] - self.boundaries[parameter][0])) + self.boundaries[parameter][0])
            elif parameter in parameters_log_uniform:
                self.pGBs[parameter] = 10**((previous_max[i] * (self.boundaries[parameter][1] - self.boundaries[parameter][0])) + self.boundaries[parameter][0])
            else:
                self.pGBs[parameter] = (previous_max[i] * (self.boundaries[parameter][1] - self.boundaries[parameter][0])) + self.boundaries[parameter][0]
            i += 1
        # self.pGBs = {'Amplitude': 4.0900673126042746e-22, 'EclipticLatitude': 0.8718477251317046, 'EclipticLongitude': 0.48599945403230693, 'Frequency': 0.003995220986111426, 'FrequencyDerivative': 1.0985841703423861e-16, 'Inclination': 1.0262955111380103, 'InitialPhase': 5.453865686076588, 'Polarization': 1.089057196561609}

        # cutoff_ratio = 1000
        # Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=self.pGBs, oversample=4, simulator="synthlisa")
        # psd_signal = np.abs(Xs.values) ** 2 + np.abs(Ys.values) ** 2 + np.abs(Zs.values) ** 2
        # highSNR = psd_signal > np.max(psd_signal) / cutoff_ratio
        # lowerindex = np.where(highSNR)[0][0] - 30
        # higherindex = np.where(highSNR)[0][-1] + 30
        # self.dataX = tdi_fs["X"].isel(f=slice(Xs.kmin, Xs.kmin + len(Xs)))[lowerindex:higherindex]
        # self.dataY = tdi_fs["Y"].isel(f=slice(Ys.kmin, Ys.kmin + len(Ys)))[lowerindex:higherindex]
        # self.dataZ = tdi_fs["Z"].isel(f=slice(Zs.kmin, Zs.kmin + len(Zs)))[lowerindex:higherindex]

        # self.DAf = (2/3*self.dataX - self.dataY - self.dataZ)/3.0
        # self.DEf = (self.dataZ - self.dataY)/np.sqrt(3.0)

        # Xs, Ys, Zs = (
        #     Xs[lowerindex:higherindex],
        #     Ys[lowerindex:higherindex],
        #     Zs[lowerindex:higherindex],
        # )

        # diff = np.abs(self.dataX - Xs.values) ** 2 + np.abs(self.dataY - Ys.values) ** 2 + np.abs(self.dataZ - Zs.values) ** 2
        # p1 = float(np.sum(diff / (self.Sn + noise)) * Xs.df) / 2.0
        # p1 = -p1
        # diff = np.abs(self.dataX) ** 2 + np.abs(self.dataY) ** 2 + np.abs(self.dataZ) ** 2
        # null_pGBs = deepcopy(self.pGBs)
        # null_pGBs['Amplitude'] = 4*10**-25
        # print('pGB', self.pGB, self.loglikelihood([self.pGB]))

    def f_statistic(self, N_frequency, N_sky):
        # We construct a global proposal density using the single
        # source F statistic to compute the individual likelihoods
        F_stat = []
        frequency = []
        eclipticlatitude = []
        eclipticlongitude = []
        pGBf = {}
        for parameter in parameters:
            pGBf[parameter] = 0
        pGBf['Amplitude'] = 1e-24
        pGBf['FrequencyDerivative'] = 0
        frequency_boundaries = [self.lower_frequency,self.upper_frequency]
        for n in range(N_sky):
            eclipticlatitude.append(self.boundaries['EclipticLatitude'][0]+(self.boundaries['EclipticLatitude'][1]-self.boundaries['EclipticLatitude'][0])*n/N_sky)
            eclipticlongitude.append(self.boundaries['EclipticLongitude'][0]+(self.boundaries['EclipticLongitude'][1]-self.boundaries['EclipticLongitude'][0])*n/N_sky)
        for k in range(N_frequency):
            F_stat.append([])
            frequency.append(frequency_boundaries[0] + (frequency_boundaries[1]-frequency_boundaries[0])*k/(N_frequency-1))
            for l in range(N_sky):
                F_stat[-1].append([])
                for m in range(N_sky):
                    F_stat[-1][-1].append(self.F_fd0(frequency[-1],eclipticlatitude[l],eclipticlongitude[m],pGBf))
        F_stat = np.asarray(F_stat)
        return F_stat, frequency, eclipticlatitude, eclipticlongitude

    def F_fd0(self, f0, theta, phi, pGBs):
        g = []
        pGBs['Frequency'] = f0
        pGBs['EclipticLongitude'] = theta
        pGBs['EclipticLatitude'] = phi
        pGBs['InitialPhase'] = 0
        pGBs['Inclination'] = np.pi/2
        pGBs['Polarization'] = 0
        g.append(self.GB.get_fd_tdixyz(template=pGBs, oversample=4, simulator="synthlisa"))
        pGBs['InitialPhase'] = np.pi
        pGBs['Inclination'] = np.pi/2
        pGBs['Polarization'] = np.pi/4
        g.append(self.GB.get_fd_tdixyz(template=pGBs, oversample=4, simulator="synthlisa"))
        pGBs['InitialPhase'] = 3*np.pi/2
        pGBs['Inclination'] = np.pi/2
        pGBs['Polarization'] = 0
        g.append(self.GB.get_fd_tdixyz(template=pGBs, oversample=4, simulator="synthlisa"))
        pGBs['InitialPhase'] = np.pi/2
        pGBs['Inclination'] = np.pi/2
        pGBs['Polarization'] = np.pi/4
        g.append(self.GB.get_fd_tdixyz(template=pGBs, oversample=4, simulator="synthlisa"))
        g2 = []
        for i in range(4):
            g2.append([])
            for j in range(3):
                g2[i].append(xr.align(self.dataX, g[i][j], join='left',fill_value=0)[1])
        g = g2
        data = [self.dataX,self.dataY,self.dataZ]
        f = 0
        for i in range(4):
            for j in range(4):
                if i != j:
                    f += 1/2* self.scalarproduct(g[i],g[j])**(-1)*self.scalarproduct(data,g[j])*self.scalarproduct(data,g[j])
        return f

    def F(self, intrinsic_parameter_values):
        g = []
        pGBs = {}
        pGBs['Amplitude'] = 1e-24
        for parameter in intrinsic_parameters:
            pGBs[parameter] = intrinsic_parameter_values[parameter]
        pGBs['InitialPhase'] = 0
        pGBs['Inclination'] = np.pi/2
        pGBs['Polarization'] = 0
        g.append(self.GB.get_fd_tdixyz(template=pGBs, oversample=4, simulator="synthlisa"))
        pGBs['InitialPhase'] = np.pi
        pGBs['Inclination'] = np.pi/2
        pGBs['Polarization'] = np.pi/4
        g.append(self.GB.get_fd_tdixyz(template=pGBs, oversample=4, simulator="synthlisa"))
        pGBs['InitialPhase'] = 3*np.pi/2
        pGBs['Inclination'] = np.pi/2
        pGBs['Polarization'] = 0
        g.append(self.GB.get_fd_tdixyz(template=pGBs, oversample=4, simulator="synthlisa"))
        pGBs['InitialPhase'] = np.pi/2
        pGBs['Inclination'] = np.pi/2
        pGBs['Polarization'] = np.pi/4
        g.append(self.GB.get_fd_tdixyz(template=pGBs, oversample=4, simulator="synthlisa"))
        g2 = []
        for i in range(4):
            g2.append([])
            for j in range(3):
                g2[i].append(xr.align(self.dataX, g[i][j], join='left',fill_value=0)[1])
        g = g2
        data = [self.dataX,self.dataY,self.dataZ]
        f = 0
        for i in range(4):
            for j in range(4):
                if i != j:
                    f += 1/2* self.scalarproduct(g[i],g[j])**(-1)*self.scalarproduct(data,g[j])*self.scalarproduct(data,g[j])
        return f

    def scalarproduct(self, a, b):
        diff = np.real(a[0].values * np.conjugate(b[0].values)) ** 2 + np.real(a[1].values * np.conjugate(b[1].values)) ** 2 + np.real(a[2].values * np.conjugate(b[2].values)) ** 2
        res = 4*float(np.sum(diff / self.Sn) * self.dataX.df)
        return res

    def plot(self, maxpGBs=None, pGBadded=None, second_data = None , found_sources_in= [], pGB_injected = [], pGB_injected_matched = [], added_label='Injection2', saving_label =None):
        plt.figure(figsize=fig_size)
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=fig_size)
        # plt.plot(dataX_training.f*1000,dataX_training.values, label='data')
        # ax1.plot(self.dataX.f * 1000, self.dataX.values.real, label="data", marker="o", zorder=5)

        # Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=self.pGB, oversample=4, simulator="synthlisa")
        # index_low = np.searchsorted(Xs.f, self.dataX.f[0])
        # Xs = Xs[index_low : index_low + len(self.dataX)]
        # Ys = Ys[index_low : index_low + len(self.dataY)]
        # Zs = Zs[index_low : index_low + len(self.dataZ)]

        # Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=self.pGB, oversample=8, simulator="synthlisa")
        # index_low = np.searchsorted(Xs.f, self.dataX.f[0])
        # Xs = Xs[index_low : index_low + len(self.dataX)]
        # Ys = Ys[index_low : index_low + len(self.dataY)]
        # Zs = Zs[index_low : index_low + len(self.dataZ)]
        # ax1.plot(Xs.f * 1000, Xs.values.real, label="VGB2", marker=".", zorder=5)
                    
        # Af = (Zs - Xs)/np.sqrt(2.0)
        ax1.plot(self.DAf.f*10**3,self.DAf,'k',zorder= 1, linewidth = 2, label = 'Data')
        ax2.plot(self.DEf.f*10**3,np.abs(self.DEf),'k',zorder= 1, linewidth = 2, label = 'Data')
        # ax1.plot(tdi_fs_long_subtracted.f[range_index],np.abs(tdi_fs_long_subtracted['X'][range_index])**2,'b',zorder= 5)

        if second_data != None:
            a,Xs = xr.align(self.dataX, second_data['X'], join='left',fill_value=0)
            a,Ys = xr.align(self.dataY, second_data['Y'], join='left',fill_value=0)
            a,Zs = xr.align(self.dataZ, second_data['Z'], join='left',fill_value=0)
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f*10**3,Af,'k--',zorder= 1, linewidth = 2, label = 'Data subtracted')
            ax2.plot(Ef.f*10**3,np.abs(Ef),'k--',zorder= 1, linewidth = 2, label = 'Data subtracted')

        for j in range(len( pGB_injected)):
            Xs, Ys, Zs = GB.get_fd_tdixyz(template= pGB_injected[j], oversample=4, simulator="synthlisa")
            a,Xs = xr.align(self.dataX, Xs, join='left',fill_value=0)
            a,Ys = xr.align(self.dataY, Ys, join='left',fill_value=0)
            a,Zs = xr.align(self.dataZ, Zs, join='left',fill_value=0)
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f*10**3,Af.data, color='grey', linewidth = 5, alpha = 0.5)
            ax2.plot(Ef.f*10**3,np.abs(Ef.data), color='grey', linewidth = 5, alpha = 0.5)

        for j in range(len(pGB_injected_matched)):
            Xs, Ys, Zs = GB.get_fd_tdixyz(template= pGB_injected_matched[j], oversample=4, simulator="synthlisa")
            a,Xs = xr.align(self.dataX, Xs, join='left',fill_value=0)
            a,Ys = xr.align(self.dataY, Ys, join='left',fill_value=0)
            a,Zs = xr.align(self.dataZ, Zs, join='left',fill_value=0)
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f*10**3,Af.data, color=colors[j%10], linewidth = 5, alpha = 0.5)
            ax2.plot(Ef.f*10**3,np.abs(Ef.data), color=colors[j%10], linewidth = 5, alpha = 0.5)


        if pGBadded != None:
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBadded, oversample=4, simulator="synthlisa")
            index_low = np.searchsorted(Xs.f, self.dataX.f[0])
            Xs = Xs[index_low : index_low + len(self.dataX)]
            Ys = Ys[index_low : index_low + len(self.dataY)]
            Zs = Zs[index_low : index_low + len(self.dataZ)]
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f* 1000, Af.data, marker='.', label=added_label)
            ax2.plot(Ef.f* 1000, np.abs(Ef.data), marker='.', label=added_label)

        for j in range(len(found_sources_in)):
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=found_sources_in[j], oversample=4, simulator="synthlisa")
            index_low = np.searchsorted(Xs.f, self.dataX.f[0])
            Xs = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
            Zs = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            Af = (Zs - Xs)/np.sqrt(2.0)
            Ef = (Zs - 2.0*Ys + Xs)/np.sqrt(6.0)
            ax1.plot(Af.f* 1000, Af.data,'--', color= colors[j%10], linewidth = 1.6)
            ax2.plot(Ef.f* 1000, np.abs(Ef.data),'--', color= colors[j%10], linewidth = 1.6)

        # ax1.plot(Xs_added2.f * 1000, Xs_added2.values.real, label="VGB2", marker=".", zorder=5)
        ax1.axvline(self.lower_frequency* 1000, color= 'red', label='Boundaries')
        ax1.axvline(self.upper_frequency* 1000, color= 'red')
        ax2.axvline(self.lower_frequency* 1000, color= 'red')
        ax2.axvline(self.upper_frequency* 1000, color= 'red')
        # ax2.axvline(self.lower_frequency* 1000- 4*32*10**-6, color= 'green')
        # ax2.axvline(self.upper_frequency* 1000+ 4*32*10**-6, color= 'green')
        # if self.reduced_frequency_boundaries != None:
        #     ax1.axvline(self.reduced_frequency_boundaries[0]* 1000, color= 'green', label='Reduced Boundaries')
        #     ax1.axvline(self.reduced_frequency_boundaries[1]* 1000, color= 'green')

        # ax1.plot(Xs.f * 1000, dataX.values.real - Xs.values.real, label="residual", alpha=0.8, color="red", marker=".")
        plt.xlabel('f (mHz)')
        ax1.set_ylabel('A')    
        ax2.set_ylabel('|E|') 
        # ax1.set_yscale('log')  
        ax2.set_yscale('log')   
        ax1.set_xlim((self.lower_frequency-self.padding)*10**3, (self.upper_frequency+self.padding)*10**3)
        ax2.set_xlim((self.lower_frequency-self.padding)*10**3, (self.upper_frequency+self.padding)*10**3)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
        # plt.legend()
        plt.pause(1)
        if saving_label != None:
            plt.savefig(saving_label,dpi=300,bbox_inches='tight')
        plt.pause(1)
        plt.show()
        # print("p true", self.loglikelihood([pGB]), "null hypothesis", self.loglikelihood([null_pGBs]))


    def SNR_split(self, pGBs):
        for i in range(len(pGBs)):
            Xs, Ys, Zs = GB.get_fd_tdixyz(template=pGBs[i], oversample=4, simulator="synthlisa")
            if i == 0:
                Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            else:
                Xs_total += xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total += xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total += xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            
        Af = (Zs_total - Xs_total)/np.sqrt(2.0)
        Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
        if self.upper_frequency > self.frequency_T_threshold:
            Tf = (Zs_total + Ys_total + Xs_total)/np.sqrt(3.0)
            hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2)/self.SA + np.absolute(Tf.data)**2 /self.ST)
            SNR2 = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA + np.real(self.DTf * np.conjugate(Tf.data))/self.ST )
        else:
            hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /self.SA)
            SNR2 = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA )
        SNR = 4.0*Xs.df* hh
        SNR2 = 4.0*Xs.df* SNR2
        SNR3 = SNR2 / np.sqrt(SNR)
        return SNR3.values

    def SNR(self, pGBs):
        for i in range(len(pGBs)):
            Xs, Ys, Zs = GB.get_fd_tdixyz(template=pGBs[i], oversample=4, simulator="synthlisa")
            if i == 0:
                Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            else:
                Xs_total += xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total += xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total += xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            
        Af = (Zs_total - Xs_total)/np.sqrt(2.0)
        Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
        if self.use_T_component:
            Tf = (Zs_total + Ys_total + Xs_total)/np.sqrt(3.0)
            hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2)/self.SA + np.absolute(Tf.data)**2 /self.ST)
            SNR2 = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA + np.real(self.DTf * np.conjugate(Tf.data))/self.ST )
        else:
            SNR2 = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA )
            hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /self.SA)
        SNR = 4.0*Xs.df* hh
        SNR2 = 4.0*Xs.df* SNR2
        SNR3 = SNR2 / np.sqrt(SNR)
        # plotIt = False
        # if plotIt:
        #     fig, ax = plt.subplots(nrows=3, sharex=True) 
        #     ax[0].plot(Af.f, np.abs(self.DAf))
        #     ax[0].plot(Af.f, np.abs(Af.data))
            
        #     ax[1].plot(Af.f, np.abs(self.DEf))
        #     ax[1].plot(Af.f, np.abs(Ef.data))
        #     ax[2].plot(Af.f, np.abs(self.DTf))
        #     ax[2].plot(Af.f, np.abs(Tf.data))
        #     plt.show()
        return SNR3.values


    def SNR_XYZ(self, pGBs):
        for i in range(len(pGBs)):
            Xs, Ys, Zs = GB.get_fd_tdixyz(template=pGBs[i], oversample=4, simulator="synthlisa")
            if i == 0:
                Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            else:
                Xs_total += xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total += xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total += xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            
        hh = np.sum((np.absolute(Xs_total.data)**2 + np.absolute(Ys_total.data)**2 + np.absolute(Zs_total.data)**2) /self.Sn)
        SNR2 = np.sum( np.real(self.dataX * np.conjugate(Xs_total.data) + self.dataY * np.conjugate(Ys_total.data)+ self.dataZ * np.conjugate(Zs_total.data))/self.Sn)
        SNR = 4.0*Xs.df* hh
        SNR2 = 4.0*Xs.df* SNR2
        SNR3 = SNR2 / np.sqrt(SNR)
        return SNR3.values

    def SNR_XYZ_Sa(self, pGBs):
        for i in range(len(pGBs)):
            Xs, Ys, Zs = GB.get_fd_tdixyz(template=pGBs[i], oversample=4, simulator="synthlisa")
            if i == 0:
                Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            else:
                Xs_total += xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total += xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total += xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            
        hh = np.sum((np.absolute(Xs_total.data)**2 + np.absolute(Ys_total.data)**2 + np.absolute(Zs_total.data)**2) /self.SA)
        SNR2 = np.sum( np.real(self.dataX * np.conjugate(Xs_total.data) + self.dataY * np.conjugate(Ys_total.data)+ self.dataZ * np.conjugate(Zs_total.data))/self.SA)
        SNR = 4.0*Xs.df* hh
        SNR2 = 4.0*Xs.df* SNR2
        SNR3 = SNR2 / np.sqrt(SNR)
        return SNR3.values

    def SNR_AE(self, pGBs):
        for i in range(len(pGBs)):
            Xs, Ys, Zs = GB.get_fd_tdixyz(template=pGBs[i], oversample=4, simulator="synthlisa")
            if i == 0:
                Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            else:
                Xs_total += xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total += xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total += xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            
        Af = (Zs_total - Xs_total)/np.sqrt(2.0)
        Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
        SNR2 = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA )
        hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /self.SA)
        SNR = 4.0*Xs.df* hh
        SNR2 = 4.0*Xs.df* SNR2
        SNR3 = SNR2 / np.sqrt(SNR)
        return SNR3.values

    def SNR2(self, pGBs):
        for i in range(len(pGBs)):
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBs[i], oversample=4, simulator="synthlisa")
            index_low = np.searchsorted(Xs.f, self.dataX.f[0])
            if i == 0:
                Xs_total = Xs[index_low : index_low + len(self.dataX)]
                Ys_total = Ys[index_low : index_low + len(self.dataY)]
                Zs_total = Zs[index_low : index_low + len(self.dataZ)]
            else:
                Xs_total += Xs[index_low : index_low + len(self.dataX)]
                Ys_total += Ys[index_low : index_low + len(self.dataY)]
                Zs_total += Zs[index_low : index_low + len(self.dataZ)]
            if len(Xs_total) < len(self.dataX):
                a,Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)
                a,Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)
                a,Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)
        Af = (Zs_total - Xs_total)/np.sqrt(2.0)
        Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)

        diff = np.abs(Af.values) ** 2 + np.abs(Ef.values) ** 2
        SNR = -float(np.sum(diff / self.SA) * self.dataX.df) /2

        return SNR#/10000

    def loglikelihoodXYZ(self, pGBs):
        for i in range(len(pGBs)):
            Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=pGBs[i], oversample=4, simulator="synthlisa")
            index_low = np.searchsorted(Xs.f, self.dataX.f[0])
            if i == 0:
                Xs_total = Xs[index_low : index_low + len(self.dataX)]
                Ys_total = Ys[index_low : index_low + len(self.dataY)]
                Zs_total = Zs[index_low : index_low + len(self.dataZ)]
            else:
                Xs_total += Xs[index_low : index_low + len(self.dataX)]
                Ys_total += Ys[index_low : index_low + len(self.dataY)]
                Zs_total += Zs[index_low : index_low + len(self.dataZ)]
            if len(Xs_total) < len(self.dataX):
                a,Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)
                a,Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)
                a,Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)

        # Af = (Zs_total - Xs_total)/np.sqrt(2.0)
        # Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
        # diff = np.abs(self.DAf - Af.values) ** 2 + np.abs(self.DEf - Ef.values) ** 2
        diff = np.abs(self.dataX - Xs_total.values) ** 2 + np.abs(self.dataY - Ys_total.values) ** 2 + np.abs(self.dataZ - Zs_total.values) ** 2
        # p1 = -float(np.sum(diff / Sn)*Xs.attrs['df'])/2.0
        p1 = float(np.sum(diff / self.Sn) * Xs_total.df) / 2.0
        # p1 = np.exp(p1)
        return -p1#/10000

    def loglikelihood(self, pGBs):
        try:
            for i in range(len(pGBs)):
                Xs, Ys, Zs = GB.get_fd_tdixyz(template=pGBs[i], oversample=4, simulator="synthlisa")
                if i == 0:
                    Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                    Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                    Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
                else:
                    Xs_total += xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                    Ys_total += xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                    Zs_total += xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
                
            Af = (Zs_total - Xs_total)/np.sqrt(2.0)
            Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
            Tf = (Zs_total + Ys_total + Xs_total)/np.sqrt(3.0)
            if self.use_T_component:
                Tf = (Zs_total + Ys_total + Xs_total)/np.sqrt(3.0)
                hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2)/self.SA + np.absolute(Tf.data)**2 /self.ST)
                SNR2 = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA + np.real(self.DTf * np.conjugate(Tf.data))/self.ST )
            else:
                SNR2 = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA )
                hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /self.SA)
            # dd = np.sum((np.absolute(self.DAf.data)**2 + np.absolute(self.DEf.data)**2) /self.SA)
            plotIt = False
            if plotIt:
                fig, ax = plt.subplots(nrows=2, sharex=True) 
                ax[0].plot(Af.f, np.abs(self.DAf))
                ax[0].plot(Af.f, np.abs(Af.data))
                
                ax[1].plot(Af.f, np.abs(self.DEf))
                ax[1].plot(Af.f, np.abs(Ef.data))
                plt.show()
            logliks = 4.0*Xs.df*( SNR2 - 0.5 * hh )
        except:
            for i in range(30):
                print('pGB error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',pGBs)
            dd = np.sum((np.absolute(self.DAf.data)**2 + np.absolute(self.DEf.data)**2) /self.SA)
            logliks = np.sqrt(4.0*Xs.df*( dd ))/1000
        return logliks.values

    def intrinsic_SNR(self, pGBs):
        for i in range(len(pGBs)):
            Xs, Ys, Zs = GB.get_fd_tdixyz(template=pGBs[i], oversample=4, simulator="synthlisa")
            if i == 0:
                Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            else:
                Xs_total += xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total += xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total += xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            
        Af = (Zs_total - Xs_total)/np.sqrt(2.0)
        Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
        if self.use_T_component:
            Tf = (Zs_total + Ys_total + Xs_total)/np.sqrt(3.0)
            hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2)/self.SA + np.absolute(Tf.data)**2 /self.ST)
        else:
            hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /self.SA)
        SNR = 4.0*Xs.df* hh
        return np.sqrt(SNR)

    def intrinsic_SNR_T(self, pGBs):
        for i in range(len(pGBs)):
            Xs, Ys, Zs = GB.get_fd_tdixyz(template=pGBs[i], oversample=4, simulator="synthlisa")
            if i == 0:
                Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            else:
                Xs_total += xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total += xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total += xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            
        Af = (Zs_total - Xs_total)/np.sqrt(2.0)
        Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
        Tf = (Zs_total + Ys_total + Xs_total)/np.sqrt(3.0)
        hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /np.absolute(Tf.data)**2)
        SNR = 4.0*Xs.df* hh
        return np.sqrt(SNR)

    def intrinsic_SNR_old(self, pGBs):
        for i in range(len(pGBs)):
            Xs, Ys, Zs = GB.get_fd_tdixyz(template=pGBs[i], oversample=4, simulator="synthlisa")
            if i == 0:
                Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            else:
                Xs_total += xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total += xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total += xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            
        Af = (Zs_total - Xs_total)/np.sqrt(2.0)
        Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
        hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /self.SA)
        SNR = 4.0*Xs.df* hh
        return np.sqrt(SNR)

    def differential_evolution_search(self, frequency_boundaries, initial_guess = None):
        bounds = []
        for signal in range(number_of_signals):
            for i in range(7):
                bounds.append((0,1))

        maxpGB = []
        self.boundaries_reduced = deepcopy(self.boundaries)
        self.boundaries_reduced['Frequency'] = frequency_boundaries
        if initial_guess != None:
            initial_guess01 = np.zeros((len(parameters)-1)*number_of_signals)
            for signal in range(number_of_signals):
                pGBstart01 = scaleto01(initial_guess[signal], self.boundaries_reduced)

                for count, parameter in enumerate(parameters_no_amplitude):
                    if pGBstart01[parameter] < 0:
                        pGBstart01[parameter] = 0
                    if pGBstart01[parameter] > 1:
                        pGBstart01[parameter] = 1
                    initial_guess01[count+(len(parameters_no_amplitude))*signal] = pGBstart01[parameter]
            start = time.time()
            res = differential_evolution(self.function_evolution, bounds=bounds, disp=False, strategy='best1exp', popsize=10,tol= 1e-8 , maxiter=1500, recombination= self.recombination, mutation=(0.5,1), x0=initial_guess01)
            print('time',time.time()-start)
        else:
            start = time.time()
            res = differential_evolution(self.function_evolution, bounds=bounds, disp=False, strategy='best1exp', popsize=8, tol= 1e-8 , maxiter=1500, recombination= self.recombination, mutation=(0.5,1))
            print('time',time.time()-start)
        for signal in range(number_of_signals):
            pGB01 = [0.5] + res.x[signal*7:signal*7+7].tolist()
            maxpGB.append(scaletooriginal(pGB01,self.boundaries_reduced))
        print(res)
        print(maxpGB)
        print('log-likelihood',self.loglikelihood(maxpGB))
        # print(pGB)
        return [maxpGB], res.nfev

    def differential_evolution_search_F(self, frequency_boundaries):
        bounds = []
        for signal in range(number_of_signals):
            for i in range(4):
                bounds.append((0,1))

        maxpGB = []
        self.boundaries_reduced = deepcopy(self.boundaries)
        self.boundaries_reduced['Frequency'] = frequency_boundaries
        start = time.time()
        res, energies = differential_evolution(self.function_evolution_F, bounds=bounds, disp=True, strategy='best1exp', popsize=5,tol= 1e-6 , maxiter=300, recombination= self.recombination, mutation=(0.5,1))
        print('time',time.time()-start)
        for signal in range(number_of_signals):
            pGB01 = [0.5] + res.x[signal*4:signal*4+4].tolist() + [0.5,0.5,0.5] 
            maxpGB.append(scaletooriginal(pGB01,self.boundaries_reduced))
        print(res)
        print(maxpGB)
        print(self.loglikelihood(maxpGB))
        # print(pGB)
        return [maxpGB], energies

    def searchCD(self):
        # np.random.seed(42)

        parameters_recorded = [None] * 10

        n_trials = 50
        number_of_evaluations = 0
        for n in range(len(parameters_recorded)):
            start = time.time()
            parameters_recorded[n] = CoordinateMC(n, self.pGBs, self.boundaries, parameters_recorded, self.SNR, n_trials=n_trials)

            print('n',n+1,'time', int(time.time()-start), np.round(parameters_recorded[n][0]['Loglikelihood'][-1],2),len(parameters_recorded[n][0]['Loglikelihood']))
            number_of_evaluations += len(parameters_recorded[n][0]['Loglikelihood']) * n_trials
        # pbar = tqdm(total=len(parameters_recorded))
        # pool = mp.Pool(mp.cpu_count())
        # parameters_recorded = pool.map(CoordinateMC, [n for n in range(len(parameters_recorded))])
        # pool.close()
        # pool.join()
        # pbar.close()

        best_run = 0
        loglikelihoodofruns = np.zeros(len(parameters_recorded))
        for i in range(len(parameters_recorded)):
            loglikelihoodofruns[i] = parameters_recorded[i][0]['Loglikelihood'][-1]
        best_value = np.max(loglikelihoodofruns)
        best_run = np.argmax(loglikelihoodofruns)
        good_runs = loglikelihoodofruns > 0
        indices = (-loglikelihoodofruns).argsort()[:len(good_runs)]
        pGBmodes = []
        for i in range(len(good_runs)):
            if good_runs[i]:
                pGBmodes.append({})
        indices = (-loglikelihoodofruns).argsort()[:len(pGBmodes)]
        pGBmodes = []
        for n in indices:
            pGBmodes.append([])
            for i in range(number_of_signals):
                pGBmodes[-1].append({})
                for parameter in parameters_no_amplitude + ['Loglikelihood']:
                    if parameter == 'Loglikelihood':
                        pGBmodes[-1][i][parameter] = parameters_recorded[n][0][parameter][-1]
                    else:
                        pGBmodes[-1][i][parameter] = parameters_recorded[n][i][parameter][-1]
        if len(pGBmodes) > 5:
            for signal in range(number_of_signals):
                pGBmodes = pGBmodes[:5]
        pGBmodes_opt = self.optimize_without_amplitude(pGBmodes)
        return [pGBmodes_opt], number_of_evaluations

    def optimize(self, pGBmodes, boundaries = None):
        if boundaries == None:
            boundaries = self.boundaries
        bounds = ()
        number_of_signals_optimize = len(pGBmodes[0])
        for signal in range(number_of_signals_optimize):
            bounds += ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1))
        for i in range(len(pGBmodes)):
            maxpGB = []
            boundaries_reduced = []
            pGBs01 = []

            # print('initial loglikelihood noise matrix', self.loglikelihood_noise_matrix(pGBmodes[0]),pGBmodes[0][0]['Frequency'])
            # print('initial loglikelihood', self.loglikelihood(pGBmodes[0]),pGBmodes[0][0]['Frequency'])
            for j in range(2):
                x = []
                for signal in range(number_of_signals_optimize):
                    if j == 0:
                        maxpGB.append({})
                        boundaries_reduced.append({})
                        for parameter in parameters:
                            maxpGB[signal][parameter] = pGBmodes[i][signal][parameter]
                    # print(maxpGB)
                    # boundaries_reduced[signal] = Reduce_boundaries(maxpGB[signal], boundaries,ratio=0.1)
                    if j > 0:
                        boundaries_reduced[signal] = Reduce_boundaries(maxpGB[signal], boundaries,ratio=0.4)
                    if j in [0]:
                        boundaries_reduced[signal] = deepcopy(boundaries)
                    pGBs01.append({})
                    for parameter in parameters:
                        if parameter in ["EclipticLatitude"]:
                            pGBs01[signal][parameter] = (np.sin(maxpGB[signal][parameter]) - boundaries_reduced[signal][parameter][0]) / (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])
                        elif parameter in ["Inclination"]:
                            pGBs01[signal][parameter] = (np.cos(maxpGB[signal][parameter]) - boundaries_reduced[signal][parameter][0]) / (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])
                        elif parameter in parameters_log_uniform:
                            pGBs01[signal][parameter] = (np.log10(maxpGB[signal][parameter]) - boundaries_reduced[signal][parameter][0]) / (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])
                        else:
                            pGBs01[signal][parameter] = (maxpGB[signal][parameter] - boundaries_reduced[signal][parameter][0]) / (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])
                    for parameter in parameters:
                        x.append(pGBs01[signal][parameter])
                # print(loglikelihood(maxpGB))
                res = scipy.optimize.minimize(self.function, x, args=boundaries_reduced, method='SLSQP', bounds=bounds, tol=1e-5)
                # res = scipy.optimize.minimize(self.function, x, args=boundaries_reduced, method='Nelder-Mead', tol=1e-10)
                # res = scipy.optimize.least_squares(self.function, x, args=boundaries_reduced, bounds=bounds)
                for signal in range(number_of_signals_optimize):
                    maxpGB[signal] = scaletooriginal(res.x[signal*8:signal*8+8],boundaries_reduced[signal])
                # print('optimized loglikelihood', loglikelihood(maxpGB),maxpGB)
                # print('boundaries reduced', boundaries_reduced)

            best_value = self.loglikelihood(maxpGB)
            if i == 0:
                current_best_value = best_value
                current_maxpGB = maxpGB
            try:
                if current_best_value < best_value:
                    current_best_value = best_value
                    current_maxpGB = maxpGB
            except:
                pass
            # print('optimized loglikelihood', self.loglikelihood(maxpGB),self.loglikelihood([self.pGB]))
        maxpGB = current_maxpGB
        # print('final optimized loglikelihood noise matrix', self.loglikelihood_noise_matrix(maxpGB),maxpGB[0]['Frequency'])
        # print('final optimized loglikelihood', self.loglikelihood(maxpGB),maxpGB[0]['Frequency'])
        return maxpGB

    def optimize_without_amplitude(self, pGBmodes, boundaries = None):
        if boundaries == None:
            boundaries = self.boundaries
        bounds = ()
        number_of_signals_optimize = len(pGBmodes[0])
        for signal in range(number_of_signals_optimize):
            bounds += ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1))
        for i in range(len(pGBmodes)):
            maxpGB = []
            pGBs01 = []

            for j in range(2):
                x = []
                for signal in range(number_of_signals_optimize):
                    if j == 0:
                        maxpGB.append({})
                        self.boundaries_reduced = {}
                        for parameter in parameters_no_amplitude:
                            maxpGB[signal][parameter] = pGBmodes[i][signal][parameter]
                    # print(maxpGB)
                    self.boundaries_reduced = Reduce_boundaries(maxpGB[signal], boundaries,ratio=0.1)
                    if j == 2:
                        self.boundaries_reduced = Reduce_boundaries(maxpGB[signal], boundaries,ratio=0.1)
                    if j in [0,1]:
                        self.boundaries_reduced = deepcopy(boundaries)
                    pGBs01.append({})
                    for parameter in parameters_no_amplitude:
                        if parameter in ["EclipticLatitude"]:
                            pGBs01[signal][parameter] = (np.sin(maxpGB[signal][parameter]) - self.boundaries_reduced[parameter][0]) / (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])
                        elif parameter in ["Inclination"]:
                            pGBs01[signal][parameter] = (np.cos(maxpGB[signal][parameter]) - self.boundaries_reduced[parameter][0]) / (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])
                        elif parameter in parameters_log_uniform:
                            pGBs01[signal][parameter] = (np.log10(maxpGB[signal][parameter]) - self.boundaries_reduced[parameter][0]) / (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])
                        else:
                            pGBs01[signal][parameter] = (maxpGB[signal][parameter] - self.boundaries_reduced[parameter][0]) / (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])
                    for parameter in parameters_no_amplitude:
                        x.append(pGBs01[signal][parameter])
                # print(loglikelihood(maxpGB))
                res = scipy.optimize.minimize(self.function_evolution, x, method='SLSQP', bounds=bounds, tol=1e-10)
                # res = scipy.optimize.minimize(self.function, x, args=self.boundaries_reduced, method='Nelder-Mead', tol=1e-10)
                # res = scipy.optimize.least_squares(self.function, x, args=self.boundaries_reduced, bounds=bounds)
                for signal in range(number_of_signals_optimize):
                    pGB01 = [0.5] + res.x[signal*7:signal*7+7].tolist()
                    maxpGB[signal] = scaletooriginal(pGB01,self.boundaries_reduced)
                    # maxpGB[signal] = scaletooriginal(res.x[signal*7:signal*7+7],self.boundaries_reduced)
                # print('optimized loglikelihood', loglikelihood(maxpGB),maxpGB)
                # print('boundaries reduced', self.boundaries_reduced)

            best_value = self.loglikelihood(maxpGB)
            if i == 0:
                current_best_value = best_value
                current_maxpGB = maxpGB
            try:
                if current_best_value < best_value:
                    current_best_value = best_value
                    current_maxpGB = maxpGB
            except:
                pass
            # print('optimized loglikelihood', self.loglikelihood(maxpGB),self.loglikelihood([self.pGB]))
        maxpGB = current_maxpGB
        print('final optimized loglikelihood', self.loglikelihood(maxpGB),maxpGB[0]['Frequency'])
        return maxpGB

    def calculate_Amplitude(self, pGBs):
        for i in range(len(pGBs)):
            Xs, Ys, Zs = GB.get_fd_tdixyz(template=pGBs[i], oversample=4, simulator="synthlisa")
            if i == 0:
                Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            else:
                Xs_total += xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total += xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total += xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
            
        Af = (Zs_total - Xs_total)/np.sqrt(2.0)
        Ef = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)
        SNR2 = np.sum( np.real(self.DAf * np.conjugate(Af.data) + self.DEf * np.conjugate(Ef.data))/self.SA )
        hh = np.sum((np.absolute(Af.data)**2 + np.absolute(Ef.data)**2) /self.SA)
        logliks = 4.0*Xs.df*( SNR2 - 0.5 * hh )
        scalar_product_hh = 4.0*Xs.df* hh
        scalar_product_dh = 4.0*Xs.df* SNR2
        A = scalar_product_dh / scalar_product_hh
        return A


    def optimizeA(self, pGBmodes, boundaries = None):
        if boundaries == None:
            boundaries = self.boundaries
        bounds = ()
        number_of_signals_optimize = len(pGBmodes[0])
        for signal in range(number_of_signals_optimize):
            bounds += ((0,1))
        for i in range(len(pGBmodes)):
            maxpGB = []
            boundaries_reduced = []
            pGBs01 = []

            for j in range(2):
                x = []
                pGBx = []
                for signal in range(number_of_signals_optimize):
                    if j == 0:
                        maxpGB.append({})
                        boundaries_reduced.append({})
                        for parameter in parameters:
                            maxpGB[signal][parameter] = pGBmodes[i][signal][parameter]
                    # print(maxpGB)
                    boundaries_reduced[signal] = Reduce_boundaries(maxpGB[signal], boundaries,ratio=0.1)
                    if j == 2:
                        boundaries_reduced[signal] = Reduce_boundaries(maxpGB[signal], boundaries,ratio=0.1)
                    if j in [0,1]:
                        boundaries_reduced[signal] = deepcopy(boundaries)
                    pGBs01.append({})
                    for parameter in parameters:
                        if parameter in ["EclipticLatitude"]:
                            pGBs01[signal][parameter] = (np.sin(maxpGB[signal][parameter]) - boundaries_reduced[signal][parameter][0]) / (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])
                        elif parameter in ["Inclination"]:
                            pGBs01[signal][parameter] = (np.cos(maxpGB[signal][parameter]) - boundaries_reduced[signal][parameter][0]) / (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])
                        elif parameter in parameters_log_uniform:
                            pGBs01[signal][parameter] = (np.log10(maxpGB[signal][parameter]) - boundaries_reduced[signal][parameter][0]) / (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])
                        else:
                            pGBs01[signal][parameter] = (maxpGB[signal][parameter] - boundaries_reduced[signal][parameter][0]) / (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])
                    for parameter in ['Amplitude']:
                        x.append(pGBs01[signal][parameter])
                    for parameter in parameters:
                        pGBx.append(pGBs01[signal][parameter])
                self.pGBx = pGBx
                res = scipy.optimize.minimize(self.functiona, x, args=(pGBx, boundaries_reduced), method='trust-constr', bounds=[bounds], tol=1e-1)
                # res = scipy.optimize.minimize(self.function, x, args=boundaries_reduced, method='Nelder-Mead', tol=1e-10)
                # res = scipy.optimize.least_squares(self.function, x, args=boundaries_reduced, bounds=bounds)
                for signal in range(number_of_signals_optimize):
                    pGB01 = deepcopy(pGBx)
                    pGB01[signal*8] = res.x[signal]
                    maxpGB[signal] = scaletooriginal(pGB01,boundaries_reduced[signal])
                # print('optimized loglikelihood', loglikelihood(maxpGB),maxpGB)
                # print('boundaries reduced', boundaries_reduced)

            best_value = self.loglikelihood(maxpGB)
            if i == 0:
                current_best_value = best_value
                current_maxpGB = maxpGB
            try:
                if current_best_value < best_value:
                    current_best_value = best_value
                    current_maxpGB = maxpGB
            except:
                pass
            # print('optimized loglikelihood', self.loglikelihood(maxpGB),self.loglikelihood([self.pGB]))
        maxpGB = current_maxpGB
        print('final optimized loglikelihood', self.loglikelihood(maxpGB),maxpGB[0]['Frequency'])
        return maxpGB

    def functionA(self, a):
        pGBs01 = self.pGBx
        boundaries_reduced = deepcopy(self.boundaries)
        pGBs = []
        for signal in range(int(len(pGBs01)/8)):
            pGBs.append({})
            i = 0
            for parameter in parameters:
                if parameter in ["EclipticLatitude"]:
                    pGBs[signal][parameter] = np.arcsin((pGBs01[signal*8:signal*8+8][i] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
                elif parameter in ["Inclination"]:
                    shifted_inclination = pGBs01[signal*8:signal*8+8][i]
                    if pGBs01[signal*8:signal*8+8][i] < 0:
                        shifted_inclination = pGBs01[signal*8:signal*8+8][i] + 1
                    if pGBs01[signal*8:signal*8+8][i] > 1:
                        shifted_inclination = pGBs01[signal*8:signal*8+8][i] - 1
                    pGBs[signal][parameter] = np.arccos((shifted_inclination * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
                elif parameter in ["FrequencyDerivative"]:
                    pGBs[signal][parameter] = 10**((pGBs01[signal*8:signal*8+8][i] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
                elif parameter in ['Amplitude']:
                    pGBs[signal][parameter] = 10**((a[0] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
                else:
                    pGBs[signal][parameter] = (pGBs01[signal*8:signal*8+8][i] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0]
                i += 1
        
        print(pGBs)
        p = -self.loglikelihood(pGBs)
        return p#/10**4

    def functiona(self,a, pGBs01, boundaries_reduced):
        pGBs = []
        for signal in range(int(len(pGBs01)/8)):
            pGBs.append({})
            i = 0
            for parameter in parameters:
                if parameter in ["EclipticLatitude"]:
                    pGBs[signal][parameter] = np.arcsin((pGBs01[signal*8:signal*8+8][i] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
                elif parameter in ["Inclination"]:
                    shifted_inclination = pGBs01[signal*8:signal*8+8][i]
                    if pGBs01[signal*8:signal*8+8][i] < 0:
                        shifted_inclination = pGBs01[signal*8:signal*8+8][i] + 1
                    if pGBs01[signal*8:signal*8+8][i] > 1:
                        shifted_inclination = pGBs01[signal*8:signal*8+8][i] - 1
                    pGBs[signal][parameter] = np.arccos((shifted_inclination * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
                elif parameter in ["FrequencyDerivative"]:
                    pGBs[signal][parameter] = 10**((pGBs01[signal*8:signal*8+8][i] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
                elif parameter in ['Amplitude']:
                    pGBs[signal][parameter] = 10**((a[0] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
                else:
                    pGBs[signal][parameter] = (pGBs01[signal*8:signal*8+8][i] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0]
                i += 1
        p = -self.loglikelihood(pGBs)
        return p#/10**4

    def function(self, pGBs01, boundaries_reduced):
        pGBs = []
        for signal in range(int(len(pGBs01)/8)):
            pGBs.append({})
            i = 0
            for parameter in parameters:
                if parameter in ["EclipticLatitude"]:
                    pGBs[signal][parameter] = np.arcsin((pGBs01[signal*8:signal*8+8][i] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
                elif parameter in ["Inclination"]:
                    shifted_inclination = pGBs01[signal*8:signal*8+8][i]
                    if pGBs01[signal*8:signal*8+8][i] < 0:
                        shifted_inclination = pGBs01[signal*8:signal*8+8][i] + 1
                    if pGBs01[signal*8:signal*8+8][i] > 1:
                        shifted_inclination = pGBs01[signal*8:signal*8+8][i] - 1
                    pGBs[signal][parameter] = np.arccos((shifted_inclination * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
                elif parameter in parameters_log_uniform:
                    pGBs[signal][parameter] = 10**((pGBs01[signal*8:signal*8+8][i] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0])
                else:
                    pGBs[signal][parameter] = (pGBs01[signal*8:signal*8+8][i] * (boundaries_reduced[signal][parameter][1] - boundaries_reduced[signal][parameter][0])) + boundaries_reduced[signal][parameter][0]
                i += 1
        p = -self.loglikelihood(pGBs)
        return p#/10**4

    def function_evolution(self, pGBs01):
        pGBs = []
        for signal in range(number_of_signals):
            pGBs.append({})
            i = 0
            for parameter in parameters:
                if parameter in ["EclipticLatitude"]:
                    pGBs[signal][parameter] = np.arcsin((pGBs01[signal*7:signal*7+7][i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
                elif parameter in ["Inclination"]:
                    pGBs[signal][parameter] = np.arccos((pGBs01[signal*7:signal*7+7][i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
                elif parameter in ['Amplitude']:
                    i -= 1
                    pGBs[signal][parameter] = 10**((0.1 * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
                # elif parameter in ["FrequencyDerivative"]:
                #     pGBs[signal][parameter] = 10**((pGBs01[signal*7:signal*7+7][i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
                else:
                    pGBs[signal][parameter] = (pGBs01[signal*7:signal*7+7][i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0]
                i += 1
        p = -self.SNR(pGBs)
        return p

    def function_evolution_F(self, pGBs01):
        pGBs =  {}
        i = 0
        for parameter in intrinsic_parameters:
            if parameter in ["EclipticLatitude"]:
                pGBs[parameter] = np.arcsin((pGBs01[i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
            elif parameter in ["Inclination"]:
                pGBs[parameter] = np.arccos((pGBs01[i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
            elif parameter in ['Amplitude']:
                i -= 1
                pGBs[parameter] = 10**((0.1 * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
            elif parameter in ["FrequencyDerivative"]:
                pGBs[parameter] = 10**((pGBs01[i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
            else:
                pGBs[parameter] = (pGBs01[i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0]
            i += 1
        intrinsic_parameter_values = {}
        for parameter in intrinsic_parameters:
            intrinsic_parameter_values[parameter] = pGBs[parameter]
        p = -self.F(intrinsic_parameter_values)
        return p

    def function_evolution_8(self, pGBs01):
        pGBs = []
        for signal in range(number_of_signals):
            pGBs.append({})
            i = 0
            for parameter in parameters:
                if parameter in ["EclipticLatitude"]:
                    pGBs[signal][parameter] = np.arcsin((pGBs01[signal*8:signal*8+8][i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
                elif parameter in ["Inclination"]:
                    pGBs[signal][parameter] = np.arccos((pGBs01[signal*8:signal*8+8][i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
                elif parameter in parameters_log_uniform:
                    pGBs[signal][parameter] = 10**((pGBs01[signal*8:signal*8+8][i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
                else:
                    pGBs[signal][parameter] = (pGBs01[signal*8:signal*8+8][i] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0]
                i += 1
        p = -self.loglikelihood(pGBs)
        return p#/10**4

    def fisher_information(self, maxpGB):
        maxpGB_changed = deepcopy(maxpGB)
        maxpGB01 = scaleto01(maxpGB, self.boundaries)
        maxpGB01_changed = deepcopy(maxpGB01)
        step_size = {}
        pGB_low = {}
        pGB_high = {}
        derivativeAf = {}
        derivativeEf = {}
        inner_product = {}
        for i in range(1):
            for parameter in parameters:
                if i == 0:
                    step_size[parameter] = 1e-9
                    # if parameter == 'Frequency':
                    #     step_size[parameter] = 0.00001
                else:
                    step_size[parameter] = 0.001/np.sqrt(inner_product[parameter][parameter])
                # if step_size[parameter] > 1e-9:
                #     step_size[parameter] = 1e-9
                pGB_low = maxpGB01[parameter] - step_size[parameter]/2
                pGB_high = maxpGB01[parameter] + step_size[parameter]/2
                # print(parameter, step_size[parameter],i)
                # print(parameter, pGB_low, pGB_high)
                if pGB_low < 0:
                    pGB_low = 0
                if pGB_high > 1:
                    pGB_high = 1
                maxpGB01_changed[parameter] = pGB_low
                maxpGB_changed = scaletooriginalparameter(maxpGB01_changed,self.boundaries)
                # print(maxpGB_changed)
                Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=maxpGB_changed, oversample=4, simulator="synthlisa")
                index_low = np.searchsorted(Xs.f, self.dataX.f[0])
                Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
                Af_low = (Zs_total - Xs_total)/np.sqrt(2.0)
                Ef_low = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)

                maxpGB01_changed[parameter] = pGB_high
                maxpGB_changed = scaletooriginalparameter(maxpGB01_changed,self.boundaries)
                Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=maxpGB_changed, oversample=4, simulator="synthlisa")
                index_low = np.searchsorted(Xs.f, self.dataX.f[0])
                Xs_total = xr.align(self.dataX, Xs, join='left',fill_value=0)[1]
                Ys_total = xr.align(self.dataY, Ys, join='left',fill_value=0)[1]
                Zs_total = xr.align(self.dataZ, Zs, join='left',fill_value=0)[1]
                Af_high = (Zs_total - Xs_total)/np.sqrt(2.0)
                Ef_high = (Zs_total - 2.0*Ys_total + Xs_total)/np.sqrt(6.0)

                derivativeAf[parameter] = (Af_high - Af_low)/step_size[parameter]
                derivativeEf[parameter] = (Ef_high - Ef_low)/step_size[parameter]

                maxpGB01_changed[parameter] = maxpGB01[parameter]

            for parameter1 in parameters:
                inner_product[parameter1] = {}
                for parameter2 in parameters:
                    AE = derivativeAf[parameter1]*np.conjugate(derivativeAf[parameter2]) + derivativeAf[parameter1]*np.conjugate(derivativeAf[parameter2])
                    inner_product[parameter1][parameter2] = 4*float(np.real(np.sum(AE / self.SA) * self.dataX.df))
            print(step_size['Amplitude'],inner_product['Amplitude']['Amplitude'],step_size['FrequencyDerivative'],inner_product['FrequencyDerivative']['FrequencyDerivative'])
        return inner_product

def objective(n,tdi_fs,Tobs):
    print('index',n)
    search = Search(n,tdi_fs,Tobs)
    # search.plot(search.pGBs)
    pGBmodes =  search.search()
    maxpGB, pGB =  search.optimize(pGBmodes)
    return maxpGB, pGB


class Global_optimizer():
    def __init__(self,tdi_fs, Tobs):
        self.tdi_fs = tdi_fs
        self.Tobs = Tobs

    def optimize(self, lower_frequency, upper_frequency, found_sources):
        # create two sets of found sources. found_sources_in with signals inside the boundary and founce_sources_out with outside sources
        found_sources_in = []
        found_sources_out = []
        for i in range(len(found_sources)):
            if found_sources[i]['Frequency'] > lower_frequency and found_sources[i]['Frequency'] < upper_frequency:
                found_sources_in.append(found_sources[i])
            else:
                found_sources_out.append(found_sources[i])

        #global optimization
        if len(found_sources_in) > 0:
            tdi_fs_subtracted = deepcopy(self.tdi_fs)
            search_out_subtracted = Search(tdi_fs_subtracted,self.Tobs, lower_frequency, upper_frequency)

            total_boundaries = deepcopy(search_out_subtracted.boundaries)
            start = time.time()
            found_sources_in = search_out_subtracted.optimize([found_sources_in], boundaries= total_boundaries)
            print('global optimization time', time.time()-start)

            found_sources = found_sources_in + found_sources_out

        return found_sources,[],[], found_sources_in, [lower_frequency, upper_frequency]

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

parameters = [
    "Amplitude",
    "EclipticLatitude",
    "EclipticLongitude",
    "Frequency",
    "FrequencyDerivative",
    "Inclination",
    "InitialPhase",
    "Polarization",
]
parameters_log_uniform = ['Amplitude']
parameters_no_amplitude = parameters[1:]
intrinsic_parameters = ['EclipticLatitude','EclipticLongitude','Frequency', 'FrequencyDerivative']

# get current directory
path = os.getcwd()
 
# parent directory
parent = os.path.dirname(path)
# grandparent directory
grandparent = os.path.dirname(parent)

DATAPATH = "/home/stefan/LDC/Radler/data"
DATAPATH = grandparent+"/LDC/Radler/data"
SAVEPATH = grandparent+"/LDC/pictures/LDC1-4/"

# sangria_fn = DATAPATH + "/dgb-tdi.h5"
# sangria_fn = DATAPATH + "/LDC1-3_VGB_v2.hdf5"
sangria_fn = DATAPATH + "/LDC1-4_GB_v2.hdf5"
# sangria_fn = DATAPATH + "/LDC1-3_VGB_v2_FD_noiseless.hdf5"
fid = h5py.File(sangria_fn)
# get the source parameters
names = np.array(fid['H5LISA/GWSources/GalBinaries'])
params = [fid['H5LISA/GWSources/GalBinaries'][k] for k in names]
reduced_names = []
i = 0
for p in params:
    i += 1
    if p.shape:
        reduced_names.append(names[i-1])
params = [np.array(p) for p in params if p.shape]
cat = np.rec.fromarrays(params, names=list(reduced_names))

# get TDI 
td = np.array(fid["H5LISA/PreProcess/TDIdata"])
td = np.rec.fromarrays(list(td.T), names=["t", "X", "Y", "Z"])
del_t = float(np.array(fid['H5LISA/GWSources/GalBinaries']['Cadence']))
reduction = 1
Tobs = float(int(np.array(fid['H5LISA/GWSources/GalBinaries']['ObservationDuration']))/reduction)

dt = del_t
# Build timeseries and frequencyseries object for X,Y,Z
tdi_ts = dict([(k, TimeSeries(td[k][:int(len(td[k][:])/reduction)], dt=dt)) for k in ["X", "Y", "Z"]])
# tdi_ts = xr.Dataset(dict([(k, TimeSeries(td[k][:int(len(td[k][:])/reduction)], dt=dt)) for k in ["X", "Y", "Z"]]))
# tdi_ts = xr.Dataset(dict([(k,TimeSeries(tdi_ts[k][:,1], dt=dt)) for k in ["X", "Y", "Z"]]))
tdi_fs = xr.Dataset(dict([(k, tdi_ts[k].ts.fft(win=window)) for k in ["X", "Y", "Z"]]))
GB = fastGB.FastGB(delta_t=dt, T=Tobs)  # in seconds

noise_model = "SciRDv1"
Nmodel = get_noise_model(noise_model, np.logspace(-5, -1, 100))

pGB = {}
ind = 0
found_sources = []
target_sources = []
first_start = time.time()
np.random.seed(42) #40
number_of_signals = 1
signals_per_subtraction = 1

f = 0.0115248
def frequency_derivative(f, Mc):
    G = 6.674*10**(-11)
    c = 3*10**8
    Mc = Mc * 2*10**30
    Mc_s = Mc*G/c**3
    return 96/(5*np.pi*Mc_s**2)*(np.pi*Mc_s*f)**(11/3)
def frequency_derivative_tyson(f):
    return 8*10**-8*f**(11/3)
def frequency_derivative_tyson_lower(f):
    return -5*10**-6*f**(13/3)
def frequency_derivative2(f, Mc_s):
    return 96/5*np.pi**(8/3)*Mc_s**(5/3)*(f)**(11/3)
print('frequency derivative range', frequency_derivative(f,0.1),frequency_derivative(f,1.4),' at f=', f)
print('frequency derivative range tyson', frequency_derivative_tyson_lower(f),frequency_derivative_tyson(f),' at f=', f)
chandrasekhar_limit = 1.4
M_chirp_upper_boundary = (chandrasekhar_limit**2)**(3/5)/(2*chandrasekhar_limit)**(1/5)

start_frequency = 0.0005
end_frequency = 0.02
number_of_windows = 0
current_frequency = deepcopy(start_frequency)
while current_frequency < end_frequency:
    current_frequency += 300*current_frequency * 10**3 / 10**9
    number_of_windows += 1

def tdi_subtraction(tdi_fs,found_sources_mp_subtract, frequencies_search):

    # found_sources_mp_best = []
    # for i in range(len(found_sources_mp_subtract)):
    #     found_sources_mp_best.append(found_sources_mp_subtract[i][0])

    # frequencies_search = np.asarray(frequencies_search)
    # found_sources_to_subtract = []
    # for i in range(len(found_sources_mp_best)):
    #     found_sources_to_subtract.append([])
    #     for j in range(len(found_sources_mp_best[i])):        
    #         # find closest frequency window
    #         frequency_window_index = np.searchsorted(frequencies_search[:,0], found_sources_mp_best[i][j]['Frequency'])-1
    #         if frequency_window_index < 0:
    #             found_sources_to_subtract[i].append(found_sources_mp_best[i][j])
    #         elif found_sources_mp_best[i][j]['Frequency'] > frequencies_search[frequency_window_index][1]:
    #             found_sources_to_subtract[i].append(found_sources_mp_best[i][j])

    #subtract the found sources from original
    tdi_fs_subtracted2 = deepcopy(tdi_fs)
    for i in range(len(found_sources_mp_subtract)):
        # for j in range(len(found_sources_to_subtract[i])):
            Xs_subtracted, Ys_subtracted, Zs_subtracted = GB.get_fd_tdixyz(template=found_sources_mp_subtract[i], oversample=4, simulator="synthlisa")
            source_subtracted = dict({"X": Xs_subtracted, "Y": Ys_subtracted, "Z": Zs_subtracted})
            index_low = np.searchsorted(tdi_fs_subtracted2["X"].f, Xs_subtracted.f[0])
            index_high = index_low+len(Xs_subtracted)
            for k in ["X", "Y", "Z"]:
                tdi_fs_subtracted2[k].data[index_low:index_high] -= source_subtracted[k].data
    return tdi_fs_subtracted2

padding = 0.5e-6

save_name = 'LDC1-4_2_years_full'
indexes = np.argsort(cat['Frequency'])
cat_sorted = cat[indexes]

# LDC1-4 #####################################
frequencies = []
frequencies_even = []
frequencies_odd = []
# search_range = [0.00398, 0.0041]
# search_range = [0.00397082, 0.00401004]
# search_range = [0.0039935, 0.0039965]
f_Nyquist = 1/dt/2
search_range = [0.0003, f_Nyquist]
# search_range = [0.0003, 0.03]
# search_range = [0.0019935, 0.0020135]
# search_range = [0.0029935, 0.0030135]
# window_length = 1*10**-7 # Hz
number_of_windows = 0
current_frequency = search_range[0]
while current_frequency < search_range[1]:
    f_smear = current_frequency *3* 10**-4
    # if current_frequency < 0.004:
    #     f_smear = current_frequency *3* 10**-4 *(1+ 3/0.004*(0.004 -current_frequency))
        # f_smear = current_frequency *3* 10**-4 *(1+ 4*np.log10(0.004 -current_frequency))
    f_deviation = frequency_derivative(current_frequency,2)*Tobs
    # window_length = np.max([f_smear, f_deviation])
    window_length = f_smear + f_deviation
    window_length += 4*32*10**-9*2
    upper_limit = current_frequency+window_length
    frequencies.append([current_frequency, upper_limit])
    current_frequency = deepcopy(upper_limit)
    number_of_windows += 1

# frequencies = frequencies[:32]
frequencies_even = frequencies[::2]
frequencies_odd = frequencies[1::2]
frequencies_search = frequencies


batch_index = 18
start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.003977)
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], cat_sorted[-2]['Frequency'])-1
# start_index = np.searchsorted(np.asarray(frequencies_search)[:,0], 0.0004)-1
batch_size = 75
# start_index= len(frequencies_search)-batch_size-1
# start_index = batch_size*batch_index
# print('batch',batch_index, start_index)
frequencies_search = frequencies_search[start_index:start_index+batch_size]
### highest + padding has to be less than f Nyqist
while frequencies_search[-1][1] + (frequencies_search[-1][1] - frequencies_search[-1][0])/2 > f_Nyquist:
    frequencies_search = frequencies_search[:-1]
# frequencies_search_modified = []
# for i in range(int(len(frequencies_search))):
#     if i % 2 == 1:
#         continue
#     frequencies_search_modified.append([frequencies_search[i][0],frequencies_search[i+1][1]])
# frequencies_search = frequencies_search_modified
# frequencies_search = frequencies_search[70:80]
# frequencies_search = frequencies_search[25:]

# target_frequencies = []
# index_low = np.searchsorted(cat_sorted['Frequency'], search_range[0])
# for i in range(10):
#     target_frequencies.append(cat_sorted[-10+i-1]['Frequency'])
#     # target_frequencies.append(cat_sorted[index_low+i*100]['Frequency'])
# # target_frequencies = cat_sorted[-17:-1]['Frequency']
# frequencies_search = []
# for i in range(len(target_frequencies)):
#     current_frequency = target_frequencies[i]
#     f_smear = current_frequency *3* 10**-4
#     f_deviation = frequency_derivative(current_frequency,M_chirp_upper_boundary)*Tobs
#     print(current_frequency,frequency_derivative(current_frequency,M_chirp_upper_boundary))
#     window_length = f_smear + f_deviation
#     window_length += 4*32*10**-9*2
#     window_shift = ((np.random.random(1)-0.5)*window_length*0.5)[0]
#     frequencies_search.append([target_frequencies[i]-window_length/2+window_shift,target_frequencies[i]+window_length/2+window_shift])

search_range = [frequencies_search[0][0],frequencies_search[-1][1]]
# search_range = [1619472*10**-8,2689639*10**-8]
print('search range '+ str(np.round(search_range[0]*10**3,4))+' mHz to '+ str(np.round(search_range[1]*10**3,4))+' mHz')

do_subtract = False
if do_subtract:
    start = time.time()
    # save_name_previous = 'found_sources397769to400619LDC1-4_4mHz_half_year_even3'
    # save_name_previous = 'found_sources397919to400770LDC1-4_4mHz_half_year_odd'
    # save_name_previous = 'found_sources397769to400619LDC1-4_4mHz_2_year_initial_half_even3'
    # save_name_previous = 'found_sources397956to401074LDC1-4_4mHz_2_year_initial_half_even_SNR10'
    # save_name_previous = 'found_sources397793to400909LDC1-4_4mHz_2_year_initial_half_odd_SNR10'
    # save_name_previous = 'found_sources397956to401074LDC1-4_4mHz_loglikelihood_ratio_threshold_even3'
    # save_name_previous = 'found_sources397793to400909LDC1-4_4mHz_loglikelihood_ratio_threshold_odd'
    # save_name_previous = 'found_sources40709to41430LDC1-4_04mHz_loglikelihood_ratio_threshold_even3'
    # save_name_previous = 'found_sources40747to41468LDC1-4_04mHz_loglikelihood_ratio_threshold_odd'
    # save_name_previous = 'LDC1-4 odd'
    # save_name_previous = 'found_sourcesLDC1-4_half_even'
    # save_name_previous = 'found_sourcesLDC1-4_half_even_T'
    # save_name_previous = 'found_sourcesLDC1-4_half_odd'
    save_name_previous = 'found_sourcesLDC1-4_2_even3'
    save_name_previous = 'found_sourcesLDC1-4_2_odd'
    found_sources_mp_subtract = np.load(SAVEPATH+''+save_name_previous+'.npy', allow_pickle = True)

    found_sources_in_flat = []
    found_sources_in_flat_frequency = []
    number_of_found_flat = 0
    for i in range(len(found_sources_mp_subtract)):
        for j in range(len(found_sources_mp_subtract[i][3])):
            found_sources_in_flat.append(found_sources_mp_subtract[i][3][j])
            found_sources_in_flat_frequency.append(found_sources_in_flat[-1]['Frequency'])
            number_of_found_flat += 1
    found_sources_in_flat_frequency = np.asarray(found_sources_in_flat_frequency)
    found_sources_in_flat = np.asarray(found_sources_in_flat)
    indexes_in = np.argsort(found_sources_in_flat_frequency)
    found_sources_in_flat_frequency = found_sources_in_flat_frequency[indexes_in]
    found_sources_in_flat = found_sources_in_flat[indexes_in]

    tdi_fs_subtracted = tdi_subtraction(tdi_fs,found_sources_in_flat, frequencies_search)
    print('subtraction time', time.time()-start)
    plot_subraction = True
    if plot_subraction:
        i = 43
        lower_frequency = frequencies_search[i][0]
        upper_frequency = frequencies_search[i][1]
        search1 = Search(tdi_fs,Tobs, lower_frequency, upper_frequency)
        search1.plot(second_data= tdi_fs_subtracted)
    tdi_fs = deepcopy(tdi_fs_subtracted)

do_print = True
if do_print:
    # found_sources_mp = np.load(SAVEPATH+'found_sources' +save_name+'.npy', allow_pickle = True)
    found_sources_mp = np.load(SAVEPATH+'found_sources' +save_name+'optimized.npy', allow_pickle = True)

    ####### change to optimized signals with neighboors removed and inside singals only
    # found_sources_mp_o = np.load(SAVEPATH+'found_sources' +save_name+'.npy', allow_pickle = True)
    # found_sources_mp_o = np.load(SAVEPATH+'found_sources' +save_name+'optimized.npy', allow_pickle = True)
    # frequencies_mp = []
    # for i in range(len(found_sources_mp_o)):
    #     frequencies_mp.append(found_sources_mp_o[i][4][0])
    # # found_sources_mp_o = np.load(SAVEPATH+'found_sources387654to408404LDC1-4_2year_even10noise_matrix.npy', allow_pickle = True)
    # found_sources_mp = np.load(SAVEPATH+'found_sources387812to408573LDC1-4_v1final_optimized.npy', allow_pickle = True)
    # # found_sources_mp = np.load(SAVEPATH+'found_sources387654to408404LDC1-4_v1final_optimized.npy', allow_pickle = True)
    # found_sources_mp_2 = deepcopy(found_sources_mp_o)
    # for i in range(len(found_sources_mp)):
    #     start_index = np.searchsorted(frequencies_mp, found_sources_mp[i][4][0])
    #     print(start_index)
    #     found_sources_mp_2[start_index][0] = found_sources_mp[i][0]
    #     found_sources_mp_2[start_index][3] = found_sources_mp[i][3]
    #     found_sources_mp_2[start_index][4] = found_sources_mp[i][4]
    # found_sources_mp = deepcopy(found_sources_mp_2)
    # np.save(SAVEPATH+'found_sources'+save_name+'optimized.npy', found_sources_mp)

    found_sources_mp_best = []
    found_sources_mp_all = []
    for i in range(len(found_sources_mp)):
        found_sources_mp_best.append(found_sources_mp[i][0])
        found_sources_in_window = []
        for j in range(len(found_sources_mp[i][1])):
            found_sources_in_window.append(found_sources_mp[i][1][j][0][0])
        found_sources_mp_all.append(found_sources_in_window)

    found_sources_in_flat = []
    found_sources_in_flat_frequency = []
    number_of_found_flat = 0
    for i in range(len(found_sources_mp)):
        for j in range(len(found_sources_mp[i][3])):
            found_sources_in_flat.append(found_sources_mp[i][3][j])
            found_sources_in_flat_frequency.append(found_sources_in_flat[-1]['Frequency'])
            number_of_found_flat += 1
    found_sources_in_flat_frequency = np.asarray(found_sources_in_flat_frequency)
    found_sources_in_flat = np.asarray(found_sources_in_flat)
    indexes_in = np.argsort(found_sources_in_flat_frequency)
    found_sources_in_flat_frequency = found_sources_in_flat_frequency[indexes_in]
    found_sources_in_flat = found_sources_in_flat[indexes_in]

    found_sources_in = []
    for i in range(len(frequencies_search)):
        lower_index = np.searchsorted(found_sources_in_flat_frequency,frequencies_search[i][0])
        higher_index = np.searchsorted(found_sources_in_flat_frequency,frequencies_search[i][1])
        if i == 102:
            print(lower_index, higher_index)
        found_sources_in.append(found_sources_in_flat[lower_index:higher_index])

    #### this way is faster if the frequency windows are the same as the loaded ones
    # found_sources_in = []
    # for i in range(len(found_sources_mp)):
    #     found_sources_in.append([])
    #     for j in range(len(found_sources_mp[i][3])):
    #         found_sources_in[i].append(found_sources_mp[i][3])

    pGB_injected = []
    for j in range(len(frequencies_search)):
        padding = (frequencies_search[j][1] - frequencies_search[j][0])/2 *0
        index_low = np.searchsorted(cat_sorted['Frequency'], frequencies_search[j][0]-padding)
        index_high = np.searchsorted(cat_sorted['Frequency'], frequencies_search[j][1]+padding)
        try:
            if cat_sorted['Frequency'][index_high] < frequencies_search[j][1]:
                index_high -= 1
        except:
            pass
        indexesA = np.argsort(-cat_sorted[index_low:index_high]['Amplitude'])
        pGB_injected_window = []
        pGB_stacked = {}
        for parameter in parameters:
            pGB_stacked[parameter] = cat_sorted[parameter][index_low:index_high][indexesA]
        for i in range(len(cat_sorted['Amplitude'][index_low:index_high])):
            pGBs = {}
            for parameter in parameters:
                pGBs[parameter] = pGB_stacked[parameter][i]
            pGB_injected_window.append(pGBs)
        pGB_injected.append(pGB_injected_window)


    # #check SNR
    # for i in range(len(found_sources_in)):
    #     if i != 1:
    #         continue
    #     print('frequency range', frequencies_search[i][0],frequencies_search[i][1])
    #     search1 = Search(tdi_fs,Tobs, frequencies_search[i][0], frequencies_search[i][1])
    #     for j in range(len( pGB_injected[i][:10])):
    #         #subtract the found sources from original
    #         tdi_fs_subtracted = deepcopy(tdi_fs)
    #         for n in range(len( pGB_injected[i][:j])):
    #             Xs_subtracted, Ys_subtracted, Zs_subtracted = GB.get_fd_tdixyz(template=pGB_injected[i][n], oversample=4, simulator="synthlisa")
    #             source_subtracted = dict({"X": Xs_subtracted, "Y": Ys_subtracted, "Z": Zs_subtracted})
    #             index_low = np.searchsorted(tdi_fs_subtracted["X"].f, Xs_subtracted.f[0])
    #             index_high = index_low+len(Xs_subtracted)
    #             for k in ["X", "Y", "Z"]:
    #                 tdi_fs_subtracted[k].data[index_low:index_high] = tdi_fs_subtracted[k].data[index_low:index_high] - source_subtracted[k].data
    #         search_subtracted = Search(tdi_fs_subtracted,Tobs, frequencies_search[i][0], frequencies_search[i][1])
    #         print('true subtracted',search_subtracted.SNR([pGB_injected[i][j]]), 'original data', search1.SNR([pGB_injected[i][j]]))
    #     for j in range(len(found_sources_in[i])):
    #         #subtract the found sources from original
    #         tdi_fs_subtracted = deepcopy(tdi_fs)
    #         for n in range(len( found_sources_in[i][:j])):
    #             Xs_subtracted, Ys_subtracted, Zs_subtracted = GB.get_fd_tdixyz(template=found_sources_in[i][n], oversample=4, simulator="synthlisa")
    #             source_subtracted = dict({"X": Xs_subtracted, "Y": Ys_subtracted, "Z": Zs_subtracted})
    #             index_low = np.searchsorted(tdi_fs_subtracted["X"].f, Xs_subtracted.f[0])
    #             index_high = index_low+len(Xs_subtracted)
    #             for k in ["X", "Y", "Z"]:
    #                 tdi_fs_subtracted[k].data[index_low:index_high] = tdi_fs_subtracted[k].data[index_low:index_high] - source_subtracted[k].data
    #         search_subtracted = Search(tdi_fs_subtracted,Tobs, frequencies_search[i][0], frequencies_search[i][1])
    #         print('found subtracted',search_subtracted.SNR([found_sources_in[i][j]]), 'original data', search1.SNR([found_sources_in[i][j]]))
    #         # print('found', search1.SNR([found_sources_mp_even_all[i][j]]))

    # number_of_found_signals = 0
    # for i in range(len(found_sources_in)):
    #     number_of_found_signals += len(found_sources_in[i])

class Posterior_computer():
    def __init__(self, tdi_fs, Tobs, frequencies, maxpGB) -> None:
        self.tdi_fs = tdi_fs
        self.Tobs = Tobs
        self.frequencies = frequencies
        self.maxpGB = maxpGB
        self.search1 = Search(self.tdi_fs,self.Tobs, self.frequencies[0], self.frequencies[1])

    def reduce_boundaries(self, plot_confidance = False):
        fisher_information_matrix = self.search1.fisher_information(self.maxpGB)
        FIM = np.zeros((len(parameters),len(parameters)))
        for i,parameter1 in enumerate(parameters):
            for j,parameter2 in enumerate(parameters):
                FIM[i,j] = fisher_information_matrix[parameter1][parameter2]
        covariance_matrix = scipy.linalg.inv(FIM)
        maxpGB01 = scaleto01(self.maxpGB, self.search1.boundaries)

        if plot_confidance:
            cov2d = np.zeros((2,2))
            index_parameter1 = parameters.index('Frequency')
            index_parameter2 = parameters.index('FrequencyDerivative')
            # index_parameter1 = parameters.index('Inclination')
            # index_parameter2 = parameters.index('Amplitude')
            index_parameter1 = parameters.index('EclipticLongitude')
            index_parameter2 = parameters.index('EclipticLatitude')
            scale_to_original = []
            scale_to_original.append((self.search1.boundaries[parameters[index_parameter1]][1] - self.search1.boundaries[parameters[index_parameter1]][0]))
            scale_to_original.append((self.search1.boundaries[parameters[index_parameter2]][1] - self.search1.boundaries[parameters[index_parameter2]][0]))
            # scale_to_original = [1,1]
            cov2d[0,0] = covariance_matrix[index_parameter1,index_parameter1] * scale_to_original[0]**2
            cov2d[0,1] = covariance_matrix[index_parameter1,index_parameter2] * scale_to_original[0] * scale_to_original[1] 
            cov2d[1,0] = covariance_matrix[index_parameter2,index_parameter1] * scale_to_original[0] * scale_to_original[1] 
            cov2d[1,1] = covariance_matrix[index_parameter2,index_parameter2] * scale_to_original[1]**2
            # cov2d = scipy.linalg.inv([[2,1],[1,1]])
            x = deepcopy(self.maxpGB[parameters[index_parameter1]])
            y = np.sin(self.maxpGB[parameters[index_parameter2]])
            samples = np.random.multivariate_normal([x,y],cov2d,10000)
            samples_x = samples[:,0]
            samples_y = samples[:,1]
            def eigsorted(cov):
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                return vals[order], vecs[:,order]
            vals, vecs = eigsorted(cov2d)
            theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            w, h = 2 * np.sqrt(vals)
            print('w',w,h)
            lambda_, v = np.linalg.eig(cov2d)
            lambda_ = np.sqrt(lambda_)
            plt.figure(figsize=[fig_size_squared[0]*0.9,fig_size_squared[1]*0.9])
            ax = plt.subplot(111, )
            for j in range(1, 4):
                ell = Ellipse(xy=(x, y),
                            width=w*j, height=h*j,
                            angle=theta, facecolor='none', edgecolor='k')
                ax.add_artist(ell)
            # confidence_ellipse(samples_x, samples_y, ax,n_std=5, edgecolor='red')
            # ax.scatter(samples_x,samples_y)
            # ax.quiver(x,y,v[0,0], v[0,1],color='r',scale = 1/lambda_[0], scale_units='xy')
            # ax.quiver(x,y,v[1,0], v[1,1],scale = 1/lambda_[1], scale_units='xy')
            # ax.scatter(x+v[0,0]*lambda_[0]*5, y+v[1,0]*lambda_[0]*5)
            # ax.scatter(x+v[0,1]*lambda_[1]*5, y+v[1,1]*lambda_[1]*5)
            ax.scatter(x, y, color='g',label='MLE')
            lbls = [r'$A$', r'$\sin \beta$', r'$\lambda$', '$f$ $($mHz$)$', r'$\log \dot{f}$ $ ($Hz/s$)$', r'$\cos \iota$', r'$\phi$', r'$\Phi$']

            plt.xlabel(lbls[index_parameter1])
            plt.ylabel(lbls[index_parameter2])
            transf = v @ np.diag(lambda_)
            scalematrix = np.max(np.abs(transf), axis=1)
            scalematrix = np.sqrt(np.diag(cov2d))
            xlim = scalematrix[0] 
            ylim = scalematrix[1] 
            print(self.search1.boundaries[parameters[index_parameter1]], xlim, scalematrix, scale_to_original, cov2d)
            # xlim = np.max(v[:,0]*lambda_[0])
            # ylim = np.max(v[:,1]*lambda_[1])
            ax.vlines([x-xlim*3,x+xlim*3],y-ylim*3,y+ylim*3,'r',label='3$\sigma$ Boundary')
            ax.hlines([y-ylim*3,y+ylim*3],x-xlim*3,x+xlim*3,'r')
            plt.xlim(x-xlim*4,x+xlim*4)
            plt.ylim(y-ylim*4,y+ylim*4)
            # plt.axis('scaled')
            plt.legend()
            plt.savefig('/home/stefan/LDC/LDC/pictures/confidance'+save_name+'.png')

        lambda_, v = scipy.linalg.eig(covariance_matrix)
        transf = np.zeros((len(lambda_),len(lambda_)))
        for i in range(len(lambda_)):
            transf[i] = v[i] * np.sqrt(lambda_[i])
        covariance2 = v @ np.diag(lambda_) @ scipy.linalg.inv(v)
        transf = v @ np.diag(np.sqrt(lambda_))
        scalematrix = np.max(np.abs(transf), axis=1)
        scalematrix = np.sqrt(np.diag(covariance_matrix))
        print('scalematrix', scalematrix)
        maxpGB01_low = deepcopy(maxpGB01)
        maxpGB01_high = deepcopy(maxpGB01)
        boundaries_reduced_fisher = {}
        sigma_multiplyer = 3
        for parameter in parameters:
            maxpGB01_low[parameter] = maxpGB01[parameter] - scalematrix[parameters.index(parameter)] * sigma_multiplyer 
            maxpGB01_high[parameter] = maxpGB01[parameter] + scalematrix[parameters.index(parameter)] * sigma_multiplyer 
            if parameter in [ 'Amplitude', 'Inclination']:
                maxpGB01_low[parameter] = maxpGB01[parameter] - scalematrix[parameters.index(parameter)] * 2
                maxpGB01_high[parameter] = maxpGB01[parameter] + scalematrix[parameters.index(parameter)] * 2
            if parameter in [ 'InitialPhase', 'Polarization']:
                maxpGB01_low[parameter] = maxpGB01[parameter] - 0.001
                maxpGB01_high[parameter] = maxpGB01[parameter] + 0.001
            if parameter in [ 'Frequency']:
                maxpGB01_low[parameter] = maxpGB01[parameter] - scalematrix[parameters.index(parameter)]  * 2
                maxpGB01_high[parameter] = maxpGB01[parameter] + scalematrix[parameters.index(parameter)] * 2
            if parameter == 'FrequencyDerivative':
                print('scale',scalematrix[parameters.index(parameter)])
                if scalematrix[parameters.index(parameter)] > 0.07:
                    print('scale', scalematrix[parameters.index(parameter)], 'fd')
                    maxpGB01_low[parameter] = maxpGB01[parameter] - 0.1
                    maxpGB01_high[parameter] = maxpGB01[parameter] + 0.1
            if maxpGB01_low[parameter] < 0:
                maxpGB01_low[parameter] = 0
            if maxpGB01_high[parameter] > 1:
                maxpGB01_high[parameter] = 1
            maxpGB_fisher_low = (maxpGB01_low[parameter] * (self.search1.boundaries[parameter][1] - self.search1.boundaries[parameter][0])) + self.search1.boundaries[parameter][0]
            maxpGB_fisher_high = (maxpGB01_high[parameter] * (self.search1.boundaries[parameter][1] - self.search1.boundaries[parameter][0])) + self.search1.boundaries[parameter][0]
            boundaries_reduced_fisher[parameter] = [maxpGB_fisher_low, maxpGB_fisher_high]

        self.boundaries_reduced = deepcopy(boundaries_reduced_fisher)
        # correct Frequency Derivative
        # split_fd = -17
        # if self.boundaries_reduced['FrequencyDerivative'][1] < split_fd+1:
        #     self.boundaries_reduced['FrequencyDerivative'][0] = -18.5
        #     self.boundaries_reduced['FrequencyDerivative'][1] = -16
        # elif self.boundaries_reduced['FrequencyDerivative'][0] < split_fd+0.5:
        #     self.boundaries_reduced['FrequencyDerivative'][0] = -18.5
        # correct Inclination and Amplitude
        split_inclination = 0.95
        if self.boundaries_reduced['Inclination'][1] > split_inclination or self.boundaries_reduced['Inclination'][0] < -split_inclination:
            if self.boundaries_reduced['Inclination'][1] > split_inclination:
                self.boundaries_reduced['Inclination'][0] = 0.5
                self.boundaries_reduced['Inclination'][1] = 1
            if self.boundaries_reduced['Inclination'][0] < -split_inclination:
                self.boundaries_reduced['Inclination'][0] = -1
                self.boundaries_reduced['Inclination'][1] = -0.5
            parameter = 'Amplitude'
            maxpGB01_low[parameter]  = maxpGB01[parameter] - 0.08
            maxpGB01_high[parameter]  = maxpGB01[parameter] + 0.08
            if maxpGB01_low[parameter] < 0:
                maxpGB01_low[parameter] = 0
            if maxpGB01_high[parameter] > 1:
                maxpGB01_high[parameter] = 1
            maxpGB_fisher_low = (maxpGB01_low[parameter] * (self.search1.boundaries[parameter][1] - self.search1.boundaries[parameter][0])) + self.search1.boundaries[parameter][0]
            maxpGB_fisher_high = (maxpGB01_high[parameter] * (self.search1.boundaries[parameter][1] - self.search1.boundaries[parameter][0])) + self.search1.boundaries[parameter][0]
            self.boundaries_reduced[parameter] = [maxpGB_fisher_low, maxpGB_fisher_high]
        print(self.boundaries_reduced)
    
    def train_model(self):
        rmse = 2
        train_size = 0
        test_size = 500
        added_trainig_size = 1000
        j = 0
        samples = np.random.rand(6000,8)
        while rmse > 0.6 and j < 5:
            j += 1
            train_size += added_trainig_size
            if j == 1:
                resolution = train_size + test_size
            else:
                resolution = added_trainig_size
            start = time.time()
            samples_likelihood = np.zeros(resolution)
            samples_likelihood2 = np.zeros(resolution)
            incl = np.zeros(resolution)
            for i in range(resolution):
                samples_p = scaletooriginal(samples[i+j*added_trainig_size], self.boundaries_reduced)
                incl[i] = samples_p['Inclination']
                samples_likelihood[i] = self.search1.loglikelihood([samples_p])
            print('sample time of', resolution, 'samples ',time.time() - start)

            samples_flat = np.zeros((resolution  , len(parameters)))
            i = 0
            boundary_ratio = 1# (boundaries_reduced1['FrequencyDerivative'][1]-boundaries_reduced1['FrequencyDerivative'][0])/(boundaries_reduced2['FrequencyDerivative'][1]-boundaries_reduced1['FrequencyDerivative'][0])
            for parameter in parameters:
                if parameter == 'FrequencyDerivative':
                    samples_flat[:, i] = samples[j*added_trainig_size:j*added_trainig_size+resolution,parameters.index(parameter)]*boundary_ratio
                else:
                    samples_flat[:, i] = samples[j*added_trainig_size:j*added_trainig_size+resolution,parameters.index(parameter)]
                i += 1
            if j == 1:
                train_y = samples_likelihood[test_size:]
                test_y = samples_likelihood[:test_size]
                train_x = samples_flat[test_size:]
                test_x = samples_flat[:test_size]
                train_x
            else:
                train_y = np.append(train_y, samples_likelihood)
                train_x = np.append(train_x, samples_flat,axis=0)

            self.mu = np.mean(train_y)
            self.sigma = np.std(train_y)
            train_y_normalized = (train_y - self.mu) / self.sigma
            kernel = RBF(length_scale=[1,2,5,1,1,1,1,1],length_scale_bounds=[(0.1,10),(0.1,10),(0.1,10),(0.1,10),(0.1,10),(0.1,10),(0.1,100),(0.1,100)])
            start = time.time()
            self.gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(train_x, train_y_normalized)
            print('train',time.time() - start)
            start = time.time()
            observed_pred_sk = self.gpr.predict(test_x)
            print('eval time of ', test_size, 'samples: ',time.time() - start)
            observed_pred_sk_scaled = observed_pred_sk*self.sigma + self.mu
            rmse = np.sqrt(mean_squared_error(test_y,observed_pred_sk_scaled))
            print("RMSE ",rmse,'with training size', len(train_y))

    def evaluate(self, x):
        partial_length = 1*10**3
        start = time.time()
        observed_pred_mean = np.zeros(len(x))
        observed_pred_sk = np.zeros(len(x))
        for n in range(int(len(x)/partial_length)):
            observed_pred_sk[n*partial_length:(n+1)*partial_length] = self.gpr.predict(x[(n)*partial_length:(n+1)*partial_length])
        try:
            observed_pred_sk[int(len(x)/partial_length)*partial_length:] = self.gpr.predict(x[int(len(x)/partial_length)*partial_length:])
        except:
            pass
        observed_pred_sk = np.asarray(observed_pred_sk)
        observed_pred_sk = observed_pred_sk.reshape(len(x))
        observed_pred_mean[:len(x)] = observed_pred_sk[:len(x)]*self.sigma + self.mu
        print('eval time', time.time()-start)
        return observed_pred_mean

    def calculate_posterior(self,resolution = 1*10**6, proposal= None, temperature = 1):
        self.resolution = resolution
        numPoints = 2**6+1
        start = time.time()
        if proposal is None:
            test_x_m = np.random.uniform(size=(self.resolution,len(parameters)))
            probability = np.ones(self.resolution)
        else:
            test_x_m = np.zeros((self.resolution,len(parameters)))
            ax = np.linspace(-0.15,1.15,numPoints)
            mypdf,axes = fastKDE.pdf(proposal[:,1],proposal[:,2], axes=[ax,ax])
            dist = Distribution(mypdf, transform=lambda i:i/len(mypdf[0,:])*(axes[0][-1]-axes[0][0])+axes[0][0])
            data, pdfs = dist(self.resolution)
            test_x_m[:,1] = data[1]
            test_x_m[:,2] = data[0]
            probability = pdfs
            mypdf,axes = fastKDE.pdf(proposal[:,0],proposal[:,5], axes=[ax,ax])
            dist = Distribution(mypdf, transform=lambda i:i/len(mypdf[0,:])*(axes[0][-1]-axes[0][0])+axes[0][0])
            data, pdfs = dist(self.resolution)
            test_x_m[:,0] = data[1]
            test_x_m[:,5] = data[0]
            # plt.figure()
            # plt.scatter(np.log10(test_x_m[:10000,0]),np.arccos(test_x_m[:10000,5]* (boundaries_reduced['Inclination'][1] - boundaries_reduced['Inclination'][0]) + boundaries_reduced['Inclination'][0]), c=pdfs[:10000])
            probability *= pdfs
            mypdf,axes = fastKDE.pdf(proposal[:,3],proposal[:,4], axes=[ax,ax])
            dist = Distribution(mypdf, transform=lambda i:i/len(mypdf[0,:])*(axes[0][-1]-axes[0][0])+axes[0][0])
            data, pdfs = dist(self.resolution)
            test_x_m[:,3] = data[1]
            test_x_m[:,4] = data[0]
            probability *= pdfs

            test_x_m[:,6] = np.random.uniform(size=self.resolution)
            test_x_m[:,7] = np.random.uniform(size=self.resolution)

            for n in range(8):
                index = np.where(test_x_m[:,n] > 0)
                test_x_m = test_x_m[index]
                probability = probability[index]
                index = np.where(test_x_m[:,n] < 1)
                test_x_m = test_x_m[index]
                probability = probability[index]

        # fig =  corner.corner(test_x_m,  bins=40, hist_kwargs={'density':True, 'lw':3}, plot_datapoints=False, fill_contours=False,  show_titles=True, \
        #                 color='#348ABD',  truth_color='k', use_math_test=True, \
        #                  levels=[0.9], title_kwargs={"fontsize": 12})
        # plt.show()

        observed_pred_mean = self.evaluate(test_x_m)
        flatsamples = np.zeros(len(test_x_m))
        flatsamplesparameters = np.zeros((len(test_x_m),len(parameters)+1))
        i = 0
        flatsamples[:] = observed_pred_mean
        flatsamplesparameters[:,1:] = test_x_m
        flatsamplesparameters[:,0] = observed_pred_mean

        maxindx = np.unravel_index(flatsamplesparameters[:,0].argmax(), flatsamplesparameters[:,0].shape)
        max_parameters = flatsamplesparameters[maxindx[0],1:]
        max_loglike = flatsamplesparameters[:,0].max()
        maxpGBpredicted = scaletooriginal(max_parameters, self.boundaries_reduced)
        if self.search1.loglikelihood([maxpGBpredicted]) > self.search1.loglikelihood([self.maxpGB]):
            maxpGB = maxpGBpredicted
            print('better maxpGB is found',maxpGB,self.search1.loglikelihood([maxpGBpredicted]), self.search1.loglikelihood([self.maxpGB]), self.maxpGB)
        best_value = self.search1.loglikelihood([self.maxpGB])
        print("pred", max_loglike, "true", self.search1.loglikelihood([scaletooriginal(max_parameters, self.boundaries_reduced)]), "max", self.search1.loglikelihood([self.maxpGB]), self.maxpGB)

        # np.random.shuffle(flatsamplesparameters)
        start = time.time()
        normalizer = np.sum(np.exp(flatsamplesparameters[:,0]-best_value))
        flatsamples_normalized = np.exp(flatsamplesparameters[:,0]-best_value)
        # flatsamples_normalized = flatsamplesparameters[:,0]
        mcmc_samples = []
        mcmc_samples.append(flatsamplesparameters[0,1:])
        previous_p = flatsamples_normalized[0]
        if previous_p == 0:
            previous_p == 1e-300
        current_parameters = flatsamplesparameters[0,1:]
        # probability = scipy.stats.multivariate_normal.pdf(flatsamplesparameters[:,1:], mean=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5], cov=[std_scale,std_scale,std_scale,std_scale,std_scale,std_scale,std_scale,std_scale])
        
        previous_probability = probability[0]
        accepted = 0
        for i in range(len(flatsamples_normalized)-1):
            if ((flatsamples_normalized[i+1] / previous_p) * (previous_probability/probability[i+1]))**(1/temperature) > np.random.uniform():
                previous_p = flatsamples_normalized[i+1]
                previous_probability = probability[i+1]
                current_parameters = flatsamplesparameters[i+1,1:]
                mcmc_samples.append(current_parameters)
                accepted += 1
            else:
                mcmc_samples.append(current_parameters)
        mcmc_samples = np.asarray(mcmc_samples)
        print('time MHMC', time.time()-start)
        print('acceptance rate %',accepted/len(probability)*100)

        return mcmc_samples

    def plot_corner(self, mcmc_samples, pGB = None, save_figure = False, save_chain = False, number_of_signal = 0, parameter_titles = False, rescaled = False):
        start = time.time()
        if not(rescaled):
            mcmc_samples_rescaled = np.zeros(np.shape(mcmc_samples))
            i = 0
            for parameter in parameters:
                if parameter in ["EclipticLatitude"]:
                    mcmc_samples_rescaled[:,i] = np.arcsin((mcmc_samples[:,parameters.index(parameter)] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
                elif parameter in ["Inclination"]:
                    mcmc_samples_rescaled[:,i] = np.arccos((mcmc_samples[:,parameters.index(parameter)] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
                elif parameter in parameters_log_uniform:
                    mcmc_samples_rescaled[:,i] = 10**((mcmc_samples[:,parameters.index(parameter)] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0])
                else:
                    mcmc_samples_rescaled[:,i] = (mcmc_samples[:,parameters.index(parameter)] * (self.boundaries_reduced[parameter][1] - self.boundaries_reduced[parameter][0])) + self.boundaries_reduced[parameter][0]
                i += 1
            print('time rescale', time.time()-start)
        else:
            mcmc_samples_rescaled = mcmc_samples
        
        save_frequency = self.maxpGB['Frequency']
        if save_chain:
            df = pd.DataFrame(data=mcmc_samples_rescaled, columns=parameters)
            df.to_csv(SAVEPATH+'Chain/frequency'+str(int(np.round(save_frequency*10**9)))+'nHz'+save_name+'.csv',index=False)
        # start = time.time()
        # df = pd.DataFrame(data=mcmc_samples_rescaled, columns=parameters)
        # df.to_csv('/home/stefan/Repositories/ldc1_evaluation_data/submission/Stefan_LDC14/GW'+str(int(np.round(maxpGB['Frequency']*10**8)))+'.csv',index=False)
        # print('saving time', time.time()-start)

        if save_figure:
            print('full time', time.time()-first_start)
            datS = np.zeros(np.shape(mcmc_samples))
            datS[:,0] = mcmc_samples_rescaled[:,2]
            datS[:,1] = np.sin(mcmc_samples_rescaled[:,1])
            datS[:,2] = mcmc_samples_rescaled[:,3]*10**3
            datS[:,3] = mcmc_samples_rescaled[:,4]
            datS[:,4] = np.cos(mcmc_samples_rescaled[:,5])
            datS[:,5] = np.log10(mcmc_samples_rescaled[:,0])
            datS[:,6] = mcmc_samples_rescaled[:,6]
            datS[:,7] = mcmc_samples_rescaled[:,7]
            lbls = [r'\lambda', r'\sin \beta', 'f$ $($mHz$)', r'\dot{f}$ $ ($Hz/s$)', r'\cos \iota', r'A', r'\phi', r'\Phi']

            if pGB != None:
                tr_s = np.zeros(len(parameters))
                maxvalues = np.zeros(len(parameters))
                i = 0
                # pGB = search1.pGB
                for parameter in ['EclipticLongitude','EclipticLatitude','Frequency','FrequencyDerivative','Inclination','Amplitude','InitialPhase','Polarization']:
                    if parameter in parameters_log_uniform:
                        tr_s[i] = np.log10(pGB[parameter])
                        maxvalues[i] = np.log10(self.maxpGB[parameter])
                    elif parameter in ['Frequency']:
                        tr_s[i] = pGB[parameter]*10**3
                        maxvalues[i] = self.maxpGB[parameter]*10**3
                    elif parameter in ['Inclination']:
                        tr_s[i] = np.cos(pGB[parameter])
                        maxvalues[i] = np.cos(self.maxpGB[parameter])
                    elif parameter in ['EclipticLatitude']:
                        tr_s[i] = np.sin(pGB[parameter])
                        maxvalues[i] = np.sin(self.maxpGB[parameter])
                    else:
                        tr_s[i] = pGB[parameter]
                        maxvalues[i] = self.maxpGB[parameter]
                    i += 1
                if tr_s[0] > np.pi:
                    tr_s[0] -= 2*np.pi

            # rng = []
            # for i in range(len(lbls)):
            #     minrange = min(datS[:,i].min(), tr_s[i])
            #     maxrange = max(datS[:,i].max(), tr_s[i])
            #     range_width = np.abs(maxrange - minrange)
            #     oner = ( minrange- range_width/10, maxrange + range_width/10)
            #     rng.append(oner)
            # Get the getdist MCSamples objects for the samples, specifying same parameter
            # names and labels; if not specified weights are assumed to all be unity
            ndim = 6
            names = ['EclipticLongitude','EclipticLatitude','Frequency','FrequencyDerivative','Inclination','Amplitude']
            labels =  lbls[:6]
            samples = MCSamples(samples=datS[:,:6],names = names, labels = labels)

            g = plots.get_subplot_plotter(subplot_size=0.9)
            samples.updateSettings({'contours': [0.68, 0.95]})
            g.settings.num_plot_contours = 2
            if parameter_titles:
                g.triangle_plot([samples], shaded=True, title_limit=2)
            else:
                g.triangle_plot([samples], shaded=True)
            
            #markers vertical
            for i in range(ndim):
                for ax in g.subplots[i:,i]:
                    # if pGB != None:
                    #     ax.axvline(tr_s[i], color='black', lw = 1)
                    ax.axvline(maxvalues[i], color='green', ls='--', lw = 1)
                i += 1
            #markers horizontal
            for i in range(ndim):
                for ax in g.subplots[i,:i]:
                    # if pGB != None:
                    #     ax.axhline(tr_s[i], color='black', lw = 1)
                    ax.axhline(maxvalues[i], color='green', ls='--', lw = 1)
                i += 1
            g.export(SAVEPATH+'Posteriors/frequency'+ str(int(np.round(save_frequency*10**9)))+save_name+str(parameter_titles)+'.png')
            # plt.show()

def compute_posterior(tdi_fs, Tobs, frequencies, maxpGB, pGB_true = None,number_of_signal = 0):
    start = time.time()
    posterior1 = Posterior_computer(tdi_fs, Tobs, frequencies, maxpGB)
    # posterior1.search1.plot(found_sources_in=[maxpGB])
    posterior1.reduce_boundaries()
    posterior1.train_model()
    mcmc_samples = posterior1.calculate_posterior(resolution = 1*10**5, temperature= 10)
    # posterior1.plot_corner(mcmc_samples, pGB_injected[1][0])
    mcmc_samples = posterior1.calculate_posterior(resolution = 1*10**5, proposal= mcmc_samples, temperature= 2)
    # posterior1.plot_corner(mcmc_samples, pGB_injected[1][0])
    mcmc_samples = posterior1.calculate_posterior(resolution = 1*10**5, proposal= mcmc_samples, temperature= 1)
    # mcmc_samples = posterior1.calculate_posterior(resolution = 1*10**5, proposal= mcmc_samples, temperature= 1)
    print('time to compute posterior: ', time.time()-start)
    posterior1.plot_corner(mcmc_samples, pGB_true, save_figure= True, save_chain= True, number_of_signal = 0, parameter_titles = False)
    return mcmc_samples



# index_of_interest_to_plot = np.searchsorted(np.asarray(frequencies_search)[:,0],  0.003716)
start = time.time()
number_of_signals = 0
frequencies_found = []
# LDC1-4 ####################
posterior_calculation_input = []
for i in range(len(found_sources_in)):
    if i != 10:
        continue
    if i in [0,len(found_sources_in)-1]:
        continue
    for j in range(len(found_sources_in[i])):
        if j != 0:
            continue
        #subtract the found sources of neighbours and own window from original except the signal itself
        tdi_fs_subtracted = deepcopy(tdi_fs)
        for m in range(3):
            # if m == 1:
            #     continue
            if i-1+m < 0:
                pass
            elif i-1+m > len(found_sources_in)-1:
                pass
            else:
                for n in range(len(found_sources_in[i-1+m])):
                    if j != n or m != 1:
                        print(i,j,m,n)
                        Xs_subtracted, Ys_subtracted, Zs_subtracted = GB.get_fd_tdixyz(template=found_sources_in[i-1+m][n], oversample=4, simulator="synthlisa")
                        source_subtracted = dict({"X": Xs_subtracted, "Y": Ys_subtracted, "Z": Zs_subtracted})
                        index_low = np.searchsorted(tdi_fs_subtracted["X"].f, Xs_subtracted.f[0])
                        index_high = index_low+len(Xs_subtracted)
                        for k in ["X", "Y", "Z"]:
                            tdi_fs_subtracted[k].data[index_low:index_high] = tdi_fs_subtracted[k].data[index_low:index_high] - source_subtracted[k].data

        search_subtracted = Search(tdi_fs_subtracted,Tobs, frequencies_search[i][0], frequencies_search[i][1])
        optimizer = Global_optimizer(tdi_fs=tdi_fs_subtracted, Tobs=Tobs)
        found_sources_mp_optimized = optimizer.optimize(frequencies_search[i][0], frequencies_search[i][1], [found_sources_in[i][j]])
        found_sources_in_optimized = found_sources_mp_optimized[3]
        print(search_subtracted.SNR([found_sources_in[i][j]]))
        print(search_subtracted.SNR([found_sources_in_optimized[0]]))

        optimizer = Global_optimizer(tdi_fs=tdi_fs_subtracted, Tobs=Tobs)
        found_sources_mp_optimized = optimizer.optimize(frequencies_search[i][0], frequencies_search[i][1], found_sources_in[i])
        found_sources_in_optimized = found_sources_mp_optimized[3]
        print(search_subtracted.SNR(found_sources_in[i]))
        print(search_subtracted.SNR(found_sources_in_optimized))

        plot_subraction = True
        if plot_subraction:
            search1 = Search(tdi_fs,Tobs, frequencies_search[i][0], frequencies_search[i][1])
            search1.plot(second_data= tdi_fs_subtracted, found_sources_in=found_sources_in[i])

        # posterior_calculation_input.append((tdi_fs_subtracted, Tobs, frequencies_search[i], found_sources_in[i][j], pGB_injected[i][j]))
        print('compute posterior of the signal',i,j, found_sources_in[i][j])
        number_of_signals += 1
        frequencies_found.append(found_sources_in[i][j]['Frequency'])
        compute_posterior(tdi_fs_subtracted, Tobs, frequencies_search[i], found_sources_in[i][j], found_sources_in[i][j])
print('time to search ', len(posterior_calculation_input), 'signals: ', time.time()-start)
print('number of signals',  len(posterior_calculation_input))
start = time.time()

#{'Amplitude': 2.031616281777476e-23, 'EclipticLatitude': -0.15997150265171872, 'EclipticLongitude': -1.6348821584004698,
# 'Frequency': 0.003979533465546961, 'FrequencyDerivative': 1.0153651382637698e-16, 'Inclination': 2.455284370386729,
#  'InitialPhase': 2.01142405115654, 'Polarization': 0.7772759959676965, 'IntrinsicSNR': 27.674888111907023}
# for i in range(len(posterior_calculation_input)):
#     for j in range(len(posterior_calculation_input)):
#         if int(np.round(posterior_calculation_input[i][3]['Frequency']*10**12)) == int(np.round(posterior_calculation_input[j][3]['Frequency']*10**12)):
#             if i != j:
#                 print(i,j,int(np.round(posterior_calculation_input[i][3]['Frequency']*10**12)))
# pool = mp.Pool(mp.cpu_count())
# number_of_threads = 16 
# pool = mp.Pool(number_of_threads)
# batch = 0
# print(len(posterior_calculation_input[batch*number_of_threads:(batch+1)*number_of_threads]))
# mcmc_samples = pool.starmap(compute_posterior,posterior_calculation_input[batch*number_of_threads:(batch+1)*number_of_threads])
# pool.close()
# pool.join()
print('time to search ', len(posterior_calculation_input), 'signals: ', time.time()-start)

