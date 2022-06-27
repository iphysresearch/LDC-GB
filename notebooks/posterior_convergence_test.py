from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
#from getdist import plots, MCSamples
import scipy
from scipy.optimize import differential_evolution
import numpy as np
import xarray as xr
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
from sources import *

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
SAVEPATH = grandparent+"/LDC/pictures/LDC1-3_v2"

# sangria_fn = DATAPATH + "/dgb-tdi.h5"
sangria_fn = DATAPATH + "/LDC1-3_VGB_v2.hdf5"
# sangria_fn = DATAPATH + "/LDC1-4_GB_v2.hdf5"
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

noise_model = "MRDv1"
Nmodel = get_noise_model(noise_model, np.logspace(-5, -1, 100))

pGB = {}
ind = 0
found_sources = []
target_sources = []
first_start = time.time()
np.random.seed(42) #40
number_of_signals = 1
signals_per_subtraction = 1

chandrasekhar_limit = 1.4
M_chirp_upper_boundary = (chandrasekhar_limit**2)**(3/5)/(2*chandrasekhar_limit)**(1/5)

start_frequency = 0.0005
end_frequency = 0.02
number_of_windows = 0
current_frequency = deepcopy(start_frequency)
while current_frequency < end_frequency:
    current_frequency += 300*current_frequency * 10**3 / 10**9
    number_of_windows += 1

padding = 0.5e-6

save_name = 'LDC1-3'
indexes = np.argsort(cat['Frequency'])
cat_sorted = cat[indexes]

# LDC1-3 ##########################################
target_frequencies = cat_sorted['Frequency']
frequencies = []
window_length = 10**-6 # Hz
for i in range(len(target_frequencies)):
    window_shift = ((np.random.random(1)-0.5)*window_length*0.5)[0]
    frequencies.append([target_frequencies[i]-window_length/2+window_shift,target_frequencies[i]+window_length/2+window_shift])
# frequencies = [frequencies[2]]
frequencies_search = frequencies
do_subtract = False

do_print = True
if do_print:
    found_sources_mp = np.load(SAVEPATH+'/found_sources' +save_name+'.npy', allow_pickle = True)
    found_sources_mp_best = []
    found_sources_mp_all = []
    frequencies_search = []
    for i in range(len(found_sources_mp)):
        found_sources_mp_best.append(found_sources_mp[i][0])
        found_sources_in_window = []
        for j in range(len(found_sources_mp[i][1])):
            found_sources_in_window.append(found_sources_mp[i][1][j][0][0])
        found_sources_mp_all.append(found_sources_in_window)
        frequencies_search.append(found_sources_mp[i][4])
        
    found_sources_in = []
    for i in range(len(found_sources_mp)):
        found_sources_in.append([])
        for j in range(len(found_sources_mp[i][3])):
            found_sources_in[i].append(found_sources_mp[i][3][j])

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


def converge_test(mcmc_samples):
    
    datS = np.zeros(np.shape(mcmc_samples))
    datS[:,0] = mcmc_samples[:,2]
    datS[:,1] = np.sin(mcmc_samples[:,1])
    datS[:,2] = mcmc_samples[:,3]*10**3
    datS[:,3] = np.log10(mcmc_samples[:,4])
    datS[:,4] = np.cos(mcmc_samples[:,5])
    datS[:,5] = np.log10(mcmc_samples[:,0])
    datS[:,6] = mcmc_samples[:,6]
    datS[:,7] = mcmc_samples[:,7]
    lbls = [r'\lambda', r'\sin \beta', 'f$ $($mHz$)', r'\log \dot{f}$ $ ($Hz/s$)', r'\cos \iota', r'A', r'\phi', r'\Phi']

    # Get the getdist MCSamples objects for the samples, specifying same parameter
    # names and labels; if not specified weights are assumed to all be unity
    names = ['EclipticLongitude','EclipticLatitude','Frequency','FrequencyDerivative','Inclination','Amplitude']
    labels =  lbls[:6]
    samples = MCSamples(samples=datS[:,:6],names = names, labels = labels)
    test_result = samples.getConvergeTests()
    return test_result

lbls = [r'\lambda', r'\sin \beta', 'f$ $($mHz$)', r'\log \dot{f}$ $ ($Hz/s$)', r'\cos \iota', r'A', r'\phi', r'\Phi']

# LDC1-3 ####################
def read_in_samples(chain_save_name_seed):
    samples = []
    for i in range(len(found_sources_in)):
        # if i != 5:
        #     continue
        for j in range(len(found_sources_in[i])):
            chain_save_name = SAVEPATH+'/Chain/frequency'+str(int(np.round(frequencies_search[i][0]*10**9)))+'nHz'+chain_save_name_seed+'.csv'
            chain_save_name = SAVEPATH+'/Chain/frequency1666286nHzLDC1-3fastGB.csv'
            df = pd.read_csv(chain_save_name)
            df['Inclination'] = np.cos(df['Inclination'].values)
            df['EclipticLatitude'] = np.sin(df['EclipticLatitude'].values)
            df['FrequencyDerivative'] = np.log10(df['FrequencyDerivative'].values)
            df['Amplitude'] = np.log10(df['Amplitude'].values)
            mcmc_samples = df.to_numpy()
            names = ['EclipticLongitude','EclipticLatitude','Frequency','FrequencyDerivative','Inclination','Amplitude','InitialPhase','Polarizatoin']
            samples.append(mcmc_samples)
    return samples

chain_save_name_seed = save_name
samples42 = read_in_samples(chain_save_name_seed)
chain_save_name_seed = save_name + 'seed41'
samples_second = read_in_samples(chain_save_name_seed)

for i in range(len(found_sources_in)):
    # if i != 5:
    #     continue
    c = ChainConsumer()
    array = np.concatenate((samples42[i][:9*10**5], samples_second[i][:9*10**5]))
    c.add_chain(array, walkers=2, name="good")
    # c.add_chain(mcmc_samples, walkers=1, name="bad")
    gelman_rubin_converged = c.diagnostic.gelman_rubin(threshold=0.003)
    geweke_converged = c.diagnostic.geweke()
    print(i,gelman_rubin_converged, geweke_converged)

import matplotlib.font_manager

injected_frequencies = []
m = 0
for i in range(len(pGB_injected)):   
    for j in range(len( pGB_injected[i])):
        injected_frequencies.append(pGB_injected[i][j]['Frequency'])
        m += 1
pGB_injected_reshaped = np.reshape(pGB_injected, m)
def closest(list, Number):
    aux = []
    for valor in list:
        aux.append(abs(Number-valor))

    return aux.index(min(aux))

lbls = [ r'\log \mathcal{A}', r'\sin \beta',r'\lambda', 'f - f_{True} $ $ ($nHz$)', '\log \dot{f} $ $ ($Hz/s$)', r'\cos \iota', r'\phi', r'\Phi']
g = plots.get_subplot_plotter(subplot_size_ratio=9/16*0.7, subplot_size=8)
g.settings.scaling_factor = 1
g.settings.line_styles = 'tab10'
g.settings.solid_colors='tab10'
boundaries = {
    "EclipticLatitude": [-1.0, 1.0],
    "EclipticLongitude": [-np.pi, np.pi],
    "Inclination": [-1.0, 1.0],
    "InitialPhase": [0.0, 2.0 * np.pi],
    "Polarization": [0.0, 1.0 * np.pi],
}
names = parameters
parameter_pairs = [['EclipticLongitude', 'EclipticLatitude'],['Inclination', 'Amplitude'],['Frequency', 'FrequencyDerivative']]
samples = []
pGB_injected_sorted_index = []
m = 0
for i in range(len(found_sources_mp)):
    for j in range(len(found_sources_in[i])):
        save_frequency = found_sources_mp[i][4][0]
        df = pd.read_csv(grandparent+"/LDC/pictures/LDC1-3_v2"+'/Chain/frequency'+str(int(np.round(save_frequency*10**9)))+'nHz'+save_name+'seed42.csv')
        df['Inclination'] = np.cos(df['Inclination'].values)
        df['EclipticLatitude'] = np.sin(df['EclipticLatitude'].values)
        df['FrequencyDerivative'] = np.log10(df['FrequencyDerivative'].values)
        df['Amplitude'] = np.log10(df['Amplitude'].values)
        pGB_injected_sorted_index.append(closest(injected_frequencies, df['Frequency'][0]))
        df['Frequency'] = (df['Frequency'] - pGB_injected_reshaped[pGB_injected_sorted_index[-1]]['Frequency'] + m*2e-9) * 1e9
        samples.append(MCSamples(samples=df.to_numpy(), names = names, labels = lbls))
        samples[-1].updateSettings({'contours': [0.68, 0.95]})
        m += 1
pGB_injected_sorted = []
for i in range(len(found_sources_mp)):
    pGB_injected_sorted.append(pGB_injected_reshaped[pGB_injected_sorted_index[i]])

g.settings.num_plot_contours = 2
# 3D (scatter) triangle plot
# you can adjust the scaling factor if font sizes are too small when
# making many subplots in a fixed size (default=2 would give smaller fonts)
g.settings.scaling_factor = 2
g.plots_2d(samples, param_pairs=parameter_pairs,legend_labels=[],lws=1.5)
for n, ax in enumerate(g.subplots[:,0]):
    parameter1, parameter2 = parameter_pairs[n]
    m = 0
    for i in range(len(pGB_injected_sorted)):   
        pGB_injected_scaled = deepcopy(pGB_injected_sorted[i])
        pGB_injected_scaled['Inclination'] = np.cos(pGB_injected_scaled['Inclination'])
        pGB_injected_scaled['EclipticLatitude'] = np.sin(pGB_injected_scaled['EclipticLatitude'])
        pGB_injected_scaled['FrequencyDerivative'] = np.log10(pGB_injected_scaled['FrequencyDerivative'])
        pGB_injected_scaled['Amplitude'] = np.log10(pGB_injected_scaled['Amplitude'])
        pGB_injected_scaled['Frequency'] = m * 2
        m += 1
        ax.plot(pGB_injected_scaled[parameter1],pGB_injected_scaled[parameter2],color='black', marker = '+',zorder=1, markersize = 10, label= 'true')
        ax.plot(pGB_injected_scaled[parameter1],pGB_injected_scaled[parameter2], marker = '+',zorder=1.1, markersize = 15,alpha = 0.5, label= 'true', linewidth = 4)
    try:
        ax.set_xlim(boundaries[parameter1])
    except:
        xlim = ax.get_xlim()
        x_length = xlim[1]-xlim[0]
        ax.set_xlim([xlim[0]-x_length*0.02, xlim[1]+x_length*0.02])
    try:
        ax.set_ylim(boundaries[parameter2])
    except:
        ylim = ax.get_ylim()
        y_length = ylim[1]-ylim[0]
        ax.set_ylim([ylim[0]-y_length*0.02, ylim[1]+y_length*0.02])
    if parameter2 in ['FrequencyDerivative']:
        ax.axhline(y=np.log10(1/Tobs**2/100), color='grey', linestyle = '--', zorder = 0.5)
        ylim = ax.get_ylim()
        y_length = ylim[1]-ylim[0]
        ax.set_ylim([-18.5, ylim[1]+y_length*0.02])

print('end')