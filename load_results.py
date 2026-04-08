import h5py
import numpy as np
import corner
import matplotlib.pyplot as plt

# Parameter order expected by the chain arrays and corner plots.
PARAM_NAMES = ['Frequency', 'FrequencyDerivative', 'Amplitude', 'RightAscension', 'Declination', 'Polarization', 'Inclination', 'InitialPhase']

# Load recovered/found sources and their matched injected counterparts.
# found_sources_fn = '/home/stefan/LDC/Mojito/found_signals/GB/found_sources_Mojito_SNR_threshold_9_seed1_overlap.h5'
found_sources_fn = '/path/to/found_signals/found_sources_Mojito_SNR_threshold_9_seed1_overlap.h5'
with h5py.File(found_sources_fn, 'r') as f:
    found_sources = f['found_sources'][:]
    injected_sources = f['injected_sources'][:]
    match_values = f['match_values'][:]


# Load one MCMC chain file for a specific group/frequency range.
chains_fn = '/path/to/chains_Mojito_SNR_threshold_9_group_1000_frequency_range_2793616nHz_to_2794522nHz.h5'
with h5py.File(chains_fn, 'r') as f:
    chains = f['chains'][:]
    initial_parameters = f['initial_parameters'][:]
    frequency_range_min = f.attrs['frequency_range_min']
    frequency_range_max = f.attrs['frequency_range_max']
# chain structure is (n_samples, n_leaves, n_params)
# n_samples is the number of samples in the chain
# n_leaves is the number of leaves in the chain, each leaf is a set of parameters for a single source
# n_params is the number of parameters in the chain

# Reorganize chain array by leaf.
# Expected shape is typically (n_samples, n_leaves, n_params), so this creates
# an array shaped (n_leaves, n_samples, n_params).
chains_leafs = []
for leaf in range(chains.shape[1]):
    chains_leafs.append(chains[:, leaf])
chains_leafs = np.array(chains_leafs)


# RJMCMC often includes extra leaves that are inactive for many samples.
# Inactive rows are all-NaN across parameters; remove those rows per leaf.
chains_leafs_cleaned = []
for leaf in range(chains_leafs.shape[0]):
    chains_leafs_cleaned.append(chains_leafs[leaf][~np.isnan(chains_leafs[leaf]).all(axis=1)])

# Plot posterior corner plots per leaf after cleaning.
for leaf in range(len(chains_leafs_cleaned)):
    fig = corner.corner(np.array(chains_leafs_cleaned[leaf]), labels=PARAM_NAMES)
    plt.show(block=True)

