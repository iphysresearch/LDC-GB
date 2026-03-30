import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import h5py
import jax
import jax.numpy as jnp
import lisaorbits
import numpy as np
import polars as pl

import corner
import matplotlib.pyplot as plt
from jaxgb.jaxgb import JaxGB

from globalGB.search_utils_GB import GB_pe, PARAM_NAMES, GBConfig
from NoiseEstimate.noise_estimate import *
from DataLoader.data_loader import LISADataLoader
from globalGB.GB_runner import GBSearchRunner

def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Search for Galactic binaries in Mojito data.")
    parser.add_argument("which_run", type=str, choices=["even1st", "even", "odd"], help="Window set to analyze.")
    parser.add_argument("batch_index", type=str, help="Batch index of frequency windows to process.")
    return parser.parse_args(argv)

# group found_sources by overlapping gw signals
def get_significant_frequency_range(source, fgb, threshold=0.1):
    """Compute the frequency range where signal amplitude > threshold * max."""
    signal = fgb.get_tdi(jnp.array(source))
    kmin = fgb.get_kmin(source[0])
    freqs = fgb.get_frequency_grid(jnp.array([kmin])).squeeze()
    max_amp = np.max(np.abs(signal[0]))
    significant_indices = np.where(np.abs(signal[0]) > max_amp * threshold)[0]
    if len(significant_indices) == 0:
        return (freqs[0], freqs[-1])
    return (freqs[significant_indices[0]], freqs[significant_indices[-1]])

def ranges_overlap(range1, range2):
    """Check if two frequency ranges overlap."""
    return range1[0] <= range2[1] and range1[1] >= range2[0]

def merge_ranges(range1, range2):
    """Merge two overlapping ranges."""
    return (min(range1[0], range2[0]), max(range1[1], range2[1]))
    
def main(argv=None):
    args = parse_args(argv)
    batch_index = int(args.batch_index)
    with open('globalGB/GB_search_config.json', 'r') as f:
        config = json.load(f)
        config = GBConfig(config)

    runner = GBSearchRunner(
        batch_index=0,
        which_run=args.which_run,
        config=config,
    )

    runner.load_data()
    runner.prepare_frequency_windows()
    savepath = runner.savepath
    with h5py.File(runner.savepath+f'/found_signals_Mojito_SNR_threshold_{int(config.snr_threshold)}_seed{config.seed}.h5', 'r') as f:
        found_sources = f['recovered_sources'][:]
    found_sources_df = pl.DataFrame(found_sources, schema=PARAM_NAMES)
    found_sources_df = found_sources_df.sort('Frequency')

    fgb = JaxGB(
    orbits=runner.waveform_args["orbits"],
    t_obs=runner.waveform_args["Tobs"],
    t0=runner.waveform_args["t0"],
    n=2**10,
    )

    try:
        with h5py.File(savepath + f'/grouped_found_sources_Mojito_SNR_threshold_{int(config.snr_threshold)}_seed{config.seed}.h5', 'r') as f:
            grouped_found_sources = [{'frequency_range': f[f'group_{i}']['frequency_range'][:], 'sources': f[f'group_{i}']['sources'][:]} for i in range(f.attrs['n_groups'])]
        print(f"Loaded {len(grouped_found_sources)} groups from {savepath + f'/grouped_found_sources_Mojito_SNR_threshold_{int(config.snr_threshold)}_seed{config.seed}.h5'}")
    except:
        print(f"No grouped found sources found at {savepath + f'/grouped_found_sources_Mojito_SNR_threshold_{int(config.snr_threshold)}_seed{config.seed}.h5'}")


        # Compute significant frequency range for each source
        ungrouped_found_sources = found_sources_df.to_numpy()
        source_ranges = []
        for i, source in enumerate(ungrouped_found_sources):
            freq_range = get_significant_frequency_range(source, fgb)
            source_ranges.append(freq_range)
            print(f"Source {i}/{len(ungrouped_found_sources)} f0={source[0]*1e3:.4f} mHz, significant range: [{freq_range[0]*1e3:.4f}, {freq_range[1]*1e3:.4f}] mHz")

        # Group sources with overlapping significant frequency ranges
        groups = []  # List of (group_freq_range, [sources])
        
        for i, source in enumerate(ungrouped_found_sources):
            print(f"Processing source {i}/{len(ungrouped_found_sources)}")
            source_range = source_ranges[i]
            merged = False
            
            for group in groups[-10:]:
                group_range, group_sources = group
                if ranges_overlap(group_range, source_range):
                    # Merge into this group
                    group[0] = merge_ranges(group_range, source_range)
                    group_sources.append(source)
                    merged = True
                    break
            
            if not merged:
                # Start a new group
                groups.append([source_range, [source]])

        # Convert to final format: list of numpy arrays
        grouped_found_sources_list = []
        for group_range, group_sources in groups:
            grouped_found_sources_list.append({
                'frequency_range': group_range,
                'sources': np.array(group_sources)
            })
            print(f"Group: {len(group_sources)} sources in range [{group_range[0]*1e3:.4f}, {group_range[1]*1e3:.4f}] mHz")

        print(f"\nTotal groups: {len(grouped_found_sources_list)}")

        # Save grouped found sources to h5 file (each group as separate dataset)
        groups_fn = savepath + f'/grouped_found_sources_Mojito_SNR_threshold_{int(config.snr_threshold)}_seed{config.seed}.h5'
        with h5py.File(groups_fn, 'w') as f:
            f.attrs['n_groups'] = len(grouped_found_sources_list)
            for i, group in enumerate(grouped_found_sources_list):
                grp = f.create_group(f'group_{i}')
                grp.create_dataset('frequency_range', data=group['frequency_range'])
                grp.create_dataset('sources', data=group['sources'])
        print(f"Saved {len(grouped_found_sources_list)} groups to {groups_fn}")

        with h5py.File(savepath + f'/grouped_found_sources_Mojito_SNR_threshold_{int(config.snr_threshold)}_seed{config.seed}.h5', 'r') as f:
            grouped_found_sources = [{'frequency_range': f[f'group_{i}']['frequency_range'][:], 'sources': f[f'group_{i}']['sources'][:]} for i in range(f.attrs['n_groups'])]
        print(f"Loaded {len(grouped_found_sources)} groups from {savepath + f'/grouped_found_sources_Mojito_SNR_threshold_{int(config.snr_threshold)}_seed{config.seed}.h5'}")

    # create dictionary with length of groups as keys and number of groups with same length as values
    group_lengths = {i: len(group['sources']) for i, group in enumerate(grouped_found_sources)}
    group_lengths_values = list(group_lengths.values())
    group_lengths_values_unique = np.unique(group_lengths_values)
    group_lengths_values_unique_counts = {int(i): int(group_lengths_values.count(i)) for i in group_lengths_values_unique}
    print(f"Group lengths: {group_lengths_values_unique_counts}")

    # index_with_most_sources = np.argmax(group_lengths_values)
    # group_with_most_sources = grouped_found_sources[index_with_most_sources]
    # print(f"Group with most sources: {index_with_most_sources}")
    # print(f"Frequency range: [{group_with_most_sources['frequency_range'][0]*1e3:.5f}, {group_with_most_sources['frequency_range'][1]*1e3:.5f}] mHz")
    # print(f"Number of sources: {len(group_with_most_sources['sources'])}")
    # print(f"{'='*60}")

    # plt.figure()
    # # plt.plot(runner.freq, np.abs(runner.tdi_fs['A']), label='A data')
    # for source in grouped_found_sources[index_with_most_sources]['sources']:
    #     As, Es, Ts = fgb.get_tdi(jnp.array(source))
    #     freq = fgb.get_frequency_grid(jnp.array([fgb.get_kmin(source[0])])).squeeze()
    #     plt.semilogy(freq, np.abs(As), label=f'Source {source[0]*1e3:.5f} mHz')
    # for source in grouped_found_sources[index_with_most_sources+1]['sources']:
    #     As, Es, Ts = fgb.get_tdi(jnp.array(source))
    #     freq = fgb.get_frequency_grid(jnp.array([fgb.get_kmin(source[0])])).squeeze()
    #     plt.semilogy(freq, np.abs(As), '--', label=f'Source {source[0]*1e3:.5f} mHz')
    # for source in grouped_found_sources[index_with_most_sources-1]['sources']:
    #     As, Es, Ts = fgb.get_tdi(jnp.array(source))
    #     freq = fgb.get_frequency_grid(jnp.array([fgb.get_kmin(source[0])])).squeeze()
    #     plt.semilogy(freq, np.abs(As), '--', label=f'Source {source[0]*1e3:.5f} mHz')
    # plt.xlabel('Frequency (mHz)')
    # plt.ylabel('|TDI A|')
    # plt.legend()
    # plt.title(f'Group {index_with_most_sources} with most sources')
    # plt.show(block=True)

    start_index = batch_index*config.batch_size
    groups_to_process = grouped_found_sources[start_index:start_index+config.batch_size]
    # Run MCMC for each group of overlapping sources
    for i, group in enumerate(groups_to_process):
        frequency_range = group['frequency_range']
        initial_parameters = group['sources']
        if len(initial_parameters) == 1:
            window_size = frequency_range[1] - frequency_range[0]
            frequency_range = [frequency_range[0]-(window_size*2), frequency_range[1]+(window_size*2)]
        print(f"Processing group {i}/{len(groups_to_process)}")
        print(f"Frequency range: [{frequency_range[0]*1e3:.4f}, {frequency_range[1]*1e3:.4f}] mHz")
        print(f"Number of sources: {len(initial_parameters)}")
        print(f"{'='*60}")
        
        gb_pe = GB_pe(
            runner.tdi_fs, 
            initial_parameters, 
            runner.Tobs, 
            frequency_range[0], 
            frequency_range[1], 
            runner.waveform_args, 
            config.dt, 
            channel_combination=config.channel_combination
        )
        start_time = time.time()
        chains, ensemble = gb_pe.mcmc_GB(nsteps=313, burn=0, ntemps=4, nwalkers=32, nleaves_max=len(initial_parameters)+2)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        chains_fn = savepath + f'/chains/chains_Mojito_SNR_threshold_{int(config.snr_threshold)}_group_{i+start_index}_frequency_range_{int(np.round(frequency_range[0]*1e9, 0))}nHz_to_{int(np.round(frequency_range[1]*1e9, 0))}nHz.h5'
        os.makedirs(os.path.dirname(chains_fn), exist_ok=True)
        with h5py.File(chains_fn, 'w') as f:
            f.create_dataset('chains', data=chains)
            f.create_dataset('initial_parameters', data=initial_parameters)
            f.attrs['frequency_range_min'] = frequency_range[0]
            f.attrs['frequency_range_max'] = frequency_range[1]
            f.attrs['time_taken'] = np.round((end_time - start_time), 0)
        print(f"Saved chains to {chains_fn}")


    # load chains
    chains_fn = savepath + f'/chains/chains_Mojito_SNR_threshold_{int(config.snr_threshold)}_group_{i+start_index}_frequency_range_{int(np.round(frequency_range[0]*1e9, 0))}nHz_to_{int(np.round(frequency_range[1]*1e9, 0))}nHz.h5'
    with h5py.File(chains_fn, 'r') as f:
        chains = f['chains'][:]
        initial_parameters = f['initial_parameters'][:]
        frequency_range_min = f.attrs['frequency_range_min']
        frequency_range_max = f.attrs['frequency_range_max']
    
    # fig = corner.corner(chains[:,0,:], labels=PARAM_NAMES)
    # plt.show(block=True)
    

if __name__ == "__main__":
    main(argv=sys.argv[1:])
