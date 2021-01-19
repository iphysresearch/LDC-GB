import matplotlib.pyplot as plt
import scipy
import numpy as np
import xarray as xr
from astropy import units as u
import pandas as pd
import time
from copy import deepcopy

import ldc.io.hdf5 as hdfio
from ldc.lisa.noise import get_noise_model
from ldc.lisa.orbits import Orbits
from ldc.lisa.projection import ProjectedStrain
from ldc.common.series import TimeSeries, FrequencySeries, window
import ldc.waveform.fastGB as fastGB
from ldc.common.tools import compute_tdi_snr
from ldc.waveform.waveform import HpHc

import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def semi_fast_tdi(config, pMBHB, t_max, dt, s_index):
    hphc = HpHc.type("MBHB-%d"%s_index, "MBHB", "IMRPhenomD")
    hphc.set_param(pMBHB)
    orbits = Orbits.type(config)
    P = ProjectedStrain(orbits)    
    yArm = P.arm_response(0, t_max, dt, [hphc], tt_order=1)
    X = P.compute_tdi_x(np.arange(0, t_max, dt))
    return TimeSeries(X, dt=dt)


# add a comment
class MLP(nn.Module):
    def __init__(self, obs_dim,):
        super(MLP, self).__init__()
        self.conv1 = nn.Conv1d(6, 32, 64)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 3, 3)
        self.fc1 = nn.Linear(141, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.shape[1]*x.shape[2])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VMBHBuffer:
    """
    Buffer to store trajectories.
    """
    def __init__(self, obs_dim, act_dim, size, gamma, lam):
        self.sources_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        # calculated TD residuals
        self.tdres_buf = np.zeros(size, dtype=np.float32)
        # rewards
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # trajectory's remaining return
        self.ret_buf = np.zeros(size, dtype=np.float32)
        # values predicted
        self.val_buf = np.zeros(size, dtype=np.float32)
        # log probabilities of chosen actions under behavior policy
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append a single timestep to the buffer. This is called at each environment
        update to store the outcome observed outcome.
        """
        # buffer has to have room so you can store
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def get(self):
        """
        Call after an epoch ends. Resets pointers and returns the buffer contents.
        """
        # Buffer has to be full before you can get something from it.
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        # TODO: Normalize the TD-residuals in self.tdres_buf
        mean = self.tdres_buf.mean()
        std = self.tdres_buf.std()
        self.tdres_buf = (self.tdres_buf-mean)/std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    tdres=self.tdres_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class Sangria:
    def __init__(self):
        DATAPATH = "/home/stefan/LDC/Sangria/data"
        sangria_fn = DATAPATH+"/dgb-tdi.h5"
        # sangria_fn = DATAPATH+"/LDC2_sangria_blind_v1.h5"
        # sangria_fn = DATAPATH+"/LDC2_sangria_gdb-tdi_v1_v3U3MxS.h5"
        # sangria_fn = DATAPATH+"/LDC2_sangria_idb-tdi_v1_DgtGV85.h5"
        # sangria_fn = DATAPATH+"/LDC2_sangria_mbhb-tdi_v1_MN5aIPz.h5"
        sangria_fn = DATAPATH+"/LDC2_sangria_training_v1.h5"
        tdi_ts, tdi_descr = hdfio.load_array(sangria_fn, name="obs/tdi")
        # sangria_fn = DATAPATH+"/LDC2_sangria_vgb-tdi_v1_sgsEVXb.h5"
        # tdi_ts, tdi_descr = hdfio.load_array(sangria_fn)
        sangria_fn_training = DATAPATH+"/LDC2_sangria_training_v1.h5"
        dt = int(1/(tdi_descr["sampling_frequency"]))

        # Build timeseries and frequencyseries object for X,Y,Z
        self.tdi_ts = xr.Dataset(dict([(k,TimeSeries(tdi_ts[k], dt=dt)) for k in ["X", "Y", "Z"]]))
        # self.tdi_ts = xr.Dataset(dict([(k,TimeSeries(tdi_ts[k][:,1], dt=dt)) for k in ["X", "Y", "Z"]]))
        self.tdi_fs = xr.Dataset(dict([(k,self.tdi_ts[k].ts.fft(win=window)) for k in ["X", "Y", "Z"]]))

        # tdi_ts_training, tdi_descr_training = hdfio.load_array(sangria_fn_training, name="obs/tdi")
        # tdi_ts_training = xr.Dataset(dict([(k,TimeSeries(tdi_ts_training[k], dt=dt)) for k in ["X", "Y", "Z"]]))
        # tdi_fs_training = xr.Dataset(dict([(k,tdi_ts_training[k].ts.fft(win=window)) for k in ["X", "Y", "Z"]]))


        noise_model = "MRDv1"
        Nmodel = get_noise_model(noise_model, np.logspace(-5, -1, 100))


        self.boundaries = {'Amplitude': [-24.0, -20.0],'EclipticLatitude': [-1.0, 1.0],
        'EclipticLongitude': [0.0, 2.0*np.pi],'Frequency': [0.0001, 0.1],'FrequencyDerivative': [-20.0, -14.0],
        'Inclination': [-1.0, 1.0],'InitialPhase': [0.0, 2.0*np.pi],'Polarization': [0.0, 2.0*np.pi]}


        vgb, units = hdfio.load_array(sangria_fn_training, name="sky/vgb/cat")
        self.GB = fastGB.FastGB(delta_t=dt, T=float(self.tdi_ts["X"].t[-1])) # in seconds
        mbhb, units = hdfio.load_array(sangria_fn, name="sky/mbhb/cat")
        config = hdfio.load_config(sangria_fn, name="obs/config")
        s_index = 0
        self.pMBHB = dict(zip(mbhb.dtype.names, mbhb[s_index]))
        t_max = float(self.tdi_ts["X"].t[-1]+self.tdi_ts["X"].attrs["dt"])
        start = time.time()
        Xs = semi_fast_tdi(config, self.pMBHB, t_max, dt, s_index)
        print(time.time()- start)

        plt.figure(figsize=(12,6))
        plt.plot(self.tdi_ts["X"].t, self.tdi_ts["X"], label="TDI X")
        plt.plot(Xs.t, (tdi_ts["X"]-Xs), label="TDI X - fast %d"%s_index)
        plt.axis([self.pMBHB["CoalescenceTime"]-1000, self.pMBHB["CoalescenceTime"]+600, None, None])
        plt.legend(loc="lower right")
        plt.xlabel("time [s]")
        plt.show()

        # Xs, Ys, Zs = self.GB.get_fd_tdixyz(template=self.pMBHB, oversample=4, simulator='synthlisa')
        self.kmin = Xs.kmin
        self.source = dict({"X":Xs, "Y":Ys, "Z":Zs})

        fmin, fmax = float(Xs.f[0]) , float(Xs.f[-1]+Xs.attrs['df'])
        f_noise = np.logspace(-5, -1, 100)
        Nmodel = get_noise_model(noise_model, f_noise)
        freq = np.array(self.source["X"].sel(f=slice(fmin, fmax)).f)
        self.Sn = Nmodel.psd(freq=freq, option='X')

        self.hid = 64  # layer width of networks
        self.l = 2  # layer number of networks
        # initialises an actor critic
        self.obs_dim = [256]
        self.parameters = ['Amplitude','EclipticLatitude','EclipticLongitude','Frequency','FrequencyDerivative','Inclination','InitialPhase','Polarization']
        self.net = {}
        for parameter in self.parameters:   
            self.net[parameter] = MLP(self.obs_dim[0])
        
    def train(self):
        """
        Main training loop.

        IMPORTANT: This function called by the checker to train your agent.
        You SHOULD NOT change the arguments this function takes and what it outputs!
        """
        

        # Training parameters
        # Number of epochs to train for
        epochs = 50
        # The longest an episode can go on before cutting it off
        batch_size = 1024
        # Learning rates for policy and value function
        lr = 3e-3

        # Set up buffer
        # buf = VMBHBuffer(self.obs_dim, act_dim, steps_per_epoch, gamma, lam)

        # Initialize the ADAM optimizer using the parameters
        # of the policy and then value networks
        # TODO: Use these optimizers later to update the policy and value networks.
        self.optimizers = {}
        for parameter in self.parameters:   
            self.optimizers[parameter] = Adam(self.net[parameter].parameters(), lr=lr)
        
        Xs = np.zeros((batch_size,self.obs_dim[0]))
        Ys = np.zeros((batch_size,self.obs_dim[0]))
        Zs = np.zeros((batch_size,self.obs_dim[0]))
        number_of_parameters = 8

        colors = plt.cm.jet(np.linspace(0,1,batch_size))
        plt.style.use('dark_background')
        # Main training loop: collect experience in env and update / log each epoch
        for epoch in range(epochs):
            pMBHBsampled = np.zeros((batch_size,number_of_parameters))
            plt.figure()
            start = time.time()
            for t in range(batch_size):
                pMBHBs = deepcopy(self.pMBHB)
                # Normal distributed proposal.
                std = np.array([5*10**-21,1, 1,1*10**-6,10**-19,1,1,1])
                # std = np.array([np.pi])
                i = 0
                for parameter in self.parameters:
                    if parameter in ['Amplitude', 'Frequency', 'FrequencyDerivative']:
                        pMBHBs[parameter] = self.pMBHB[parameter]+(np.random.rand()-0.5)*std[i]
                    else:
                        pMBHBs[parameter] = np.random.uniform(self.boundaries[parameter][0], self.boundaries[parameter][1])
                    if parameter == 'Amplitude':
                        pMBHBsampled[t,i] = np.log10(pMBHBs[parameter])
                    elif parameter == 'Frequency':
                        pMBHBsampled[t,i] = pMBHBs[parameter]*10**4
                    else:
                        pMBHBsampled[t,i] = pMBHBs[parameter]
                    i += 1
                # for parameter in ['EclipticLatitude']:
                #     pMBHBs[parameter] = self.pMBHB[parameter]+t/batch_size*std[i]
                #     pMBHBsampled[t,i] = self.pMBHB[parameter]+t/batch_size*std[i]
                #     i += 1
                # print(self.pMBHB)
                # print(pMBHBsampled)
                Xb, Yb, Zb = self.GB.get_fd_tdixyz(template=pMBHBs, oversample=4, simulator='synthlisa')
                Xb.values, Yb.values, Zb.values = Xb.values*10**18, Yb.values*10**18, Zb.values*10**18
                plt.plot(Xb.f*1000,Xb.values, label='binary', color=colors[t], alpha = 1)

                if (Xb.kmin-self.kmin) >= 0:
                    try:
                        Xs[t,(Xb.kmin-self.kmin):(Xb.kmin-self.kmin)+len(Xb)] = Xb[:]
                        Ys[t,(Yb.kmin-self.kmin):(Yb.kmin-self.kmin)+len(Yb)] = Yb[:]
                        Zs[t,(Zb.kmin-self.kmin):(Zb.kmin-self.kmin)+len(Zb)] = Zb[:]
                    except:
                        Xs[t,(Xb.kmin-self.kmin):] = Xb[:self.obs_dim[0]-(Xb.kmin-self.kmin)]
                        Ys[t,(Yb.kmin-self.kmin):] = Yb[:self.obs_dim[0]-(Yb.kmin-self.kmin)]
                        Zs[t,(Zb.kmin-self.kmin):] = Zb[:self.obs_dim[0]-(Zb.kmin-self.kmin)]
                else:
                    try:
                        Xs[t,:] = Xb[-(Xb.kmin-self.kmin):-(Xb.kmin-self.kmin)+self.obs_dim[0]]
                        Ys[t,:] = Yb[-(Yb.kmin-self.kmin):-(Yb.kmin-self.kmin)+self.obs_dim[0]]
                        Zs[t,:] = Zb[-(Zb.kmin-self.kmin):-(Zb.kmin-self.kmin)+self.obs_dim[0]]
                    except:
                        Xs[t,:(Xb.kmin-self.kmin)+len(Xb)] = Xb[-(Xb.kmin-self.kmin):]
                        Ys[t,:(Yb.kmin-self.kmin)+len(Yb)] = Yb[-(Yb.kmin-self.kmin):]
                        Zs[t,:(Zb.kmin-self.kmin)+len(Zb)] = Zb[-(Zb.kmin-self.kmin):]
            print(time.time()- start)   
            Xsr = torch.tensor(Xs.real).float()
            Xsi = torch.tensor(Xs.imag).float()
            Ysr = torch.tensor(Ys.real).float()
            Ysi = torch.tensor(Ys.imag).float()
            Zsr = torch.tensor(Zs.real).float()
            Zsi = torch.tensor(Zs.imag).float()
            input_data = torch.tensor([Xs.real,Xs.imag,Ys.real,Ys.imag,Zs.real,Zs.imag])
            input_data = torch.reshape(input_data,(batch_size,6,self.obs_dim[0])).float()
            pMBHBsampled = torch.tensor(pMBHBsampled).float()   
            plt.figure()
            plt.imshow(Xs)
            plt.show()

            print(f"Epoch: {epoch+1}/{epochs}")
            # This is the end of an epoch, so here is where you likely want to update
            # the policy and / or value function.
            # TODO: Implement the polcy and value function update. Hint: some of the torch code is
            # done for you.
            criterion = torch.nn.MSELoss()
            i = 0
            for parameter in self.parameters:
                result = self.net[parameter].forward(input_data)
                loss_prev = criterion(result,pMBHBsampled[:,i])
                for _ in range(100):
                    self.optimizers[parameter].zero_grad()
                    #compute a loss for the value function, call loss.backwards() and then
                    # pi_total, log_prob = self.ac.pi.forward(data['obs'],act=data['act'])
                    # loss_v = torch.sum(torch.mul(data["tdres"].detach(),log_prob))/len(ep_returns)
                    result = self.net[parameter].forward(input_data)
                    loss = criterion(result,pMBHBsampled[:,i])

                    loss.backward()
                    # print(loss_v)
                    # print(list(self.ac.v.parameters())[-1].grad)
                    self.optimizers[parameter].step()
                # print(result - pMBHBsampled[:,i])
                print(parameter,loss.sqrt(),loss_prev.sqrt(), result.view(-1, result.shape[1]*result.shape[0]),pMBHBsampled[:,i])
                i += 1
            # print(self.std_ret)

        return True


    def get_action(self, obs):
        """
        Sample an action from your policy.

        IMPORTANT: This function called by the checker to evaluate your agent.
        You SHOULD NOT change the arguments this function takes and what it outputs!
        """
        # TODO: Implement this function.
        # Currently, this just returns a random action.
        obs = torch.tensor(obs)
        pi_total, _ = self.ac.pi.forward(obs)
        action = int(pi_total.probs.argmax())
        # print(action)
        return action


def main():
    """
    Train and evaluate agent.

    This function basically does the same as the checker that evaluates your agent.
    You can use it for debugging your agent and visualizing what it does.
    """


    AI = Sangria()
    AI.train()

    episode_length = 300
    n_eval = 100
    returns = []
    print("Evaluating agent...")

    for i in range(n_eval):
        print(f"Testing policy: episode {i+1}/{n_eval}")
        state = env.reset()
        cumulative_return = 0
        # The environment will set terminal to True if an episode is done.
        terminal = False
        env.reset()
        for t in range(episode_length):
            if i <= 10:
                rec.capture_frame()
            # Taking an action in the environment
            action = agent.get_action(state)
            state, reward, terminal = env.transition(action)
            cumulative_return += reward
            if terminal:
                break
        returns.append(cumulative_return)
        print(f"Achieved {cumulative_return:.2f} return.")
        if i == 10:
            rec.close()
            print("Saved video of 10 episodes to 'policy.mp4'.")
    env.close()
    print(f"Average return: {np.mean(returns):.2f}")

if __name__ == "__main__":
    main()
