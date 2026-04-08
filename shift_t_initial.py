import h5py
import numpy as np
import corner
import matplotlib.pyplot as plt
import os

import jax
import jax.numpy as jnp

from jaxgb.jaxgb import JaxGB
from jaxgb.params import GBObject

jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)

def shift_to_tinit(found_sources: np.ndarray, t0: float, t_init: float) -> np.ndarray:
    """Shift found sources to the correct initial time."""
    # shift found sources to the same initial time
    gbo = GBObject.from_jaxgb_params(jnp.array(found_sources), t_init=t0)
    found_sources_t_init = gbo.to_jaxgb_array(t0=t_init)
    return found_sources_t_init

t0 = 98361099.827664
t_init = 97729089.327664 


# load chains from directory
chains_dir = '/home/stefan/LDC/Mojito/found_signals/GB/chains_t0'
for i, chains_fn in enumerate(os.listdir(chains_dir)):
    print(f'Processing {i}/{len(os.listdir(chains_dir))}')
    with h5py.File(os.path.join(chains_dir, chains_fn), 'r') as f:
        chains = f['chains'][:]
        initial_parameters = f['initial_parameters'][:]
        frequency_range_min = f.attrs['frequency_range_min']
        frequency_range_max = f.attrs['frequency_range_max']



    chains_t_init = []
    for leaf in range(chains.shape[1]):
        chains_t_init.append(shift_to_tinit(chains[:, leaf], t0, t_init))
    initial_parameters_t_init = shift_to_tinit(initial_parameters, t0, t_init)
    # swap axes of chains_t_init
    chains_t_init = jnp.array(chains_t_init).swapaxes(0, 1)

    # save chains to h5 file
    chains_fn = os.path.join(chains_dir[:-9], 'chains_t_init', chains_fn)
    os.makedirs(os.path.dirname(chains_fn), exist_ok=True)
    with h5py.File(chains_fn, 'w') as f:
        f.create_dataset('chains', data=chains_t_init)
        f.create_dataset('initial_parameters', data=initial_parameters_t_init)
        f.attrs['frequency_range_min'] = frequency_range_min
        f.attrs['frequency_range_max'] = frequency_range_max
        f.attrs['t0'] = t0
        f.attrs['t_init'] = t_init

