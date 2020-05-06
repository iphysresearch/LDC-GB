# distutils: sources = GB.c LISA.c 
# distutils: language = c
# cython: language_level=3

cdef extern from "GB.h":
    void Fast_GB(double* , long, double, double,  double*, double*, double*, double*, double*, double*, int);
        

import numpy as np
cimport numpy as np
from ldc.common import constants
from ldc.lisa.noise import simple_snr
import math


# TODO:
# orbits: use ldc.orbits in lisa.c
# parameters: check pycbc conventions, give info on expected units, getter/setter tools
# time domain
# output as frequency array


YEAR = constants.Nature.SIDEREALYEAR_J2000DAY*24*60*60

cdef class pyGB:
    cdef public double arm_length
    cdef public long M,N
    cdef public double f0,fdot,ampl,theta,phi,psi,incl,phi0
    cdef public double T, delta_t
    cdef public int oversample
    cdef public int kmin
    
    def __cinit__(self, orbits=None, T=6.2914560e7, delta_t=15):
        """ Define C++ FastBinary dimensions and check that orbits are
        compatible.
        """
        if orbits is not None:
            if not isinstance(orbits, "AnalyticOrbits"):
                raise TypeError('Fastbinary approximation requires analytic orbits')
            else:
                self.arm_length = orbits.arm_length
                if orbits.initial_rotation !=0 or orbits.initial_position !=0:
                    raise ValueError('Fastbinary approximation requires null initial rotation and position')
        else:
            self.arm_length = 2.5e9
        self.T, self.delta_t = T, delta_t
    
    def buffersize(self, f0, ampl, oversample):
        """Get array dimension needed to compute TDI. 
        """
        Acut = simple_snr(f0,ampl,years=self.T/YEAR)
        mult = 8
        if((self.T/YEAR) <= 8.0): mult = 8
        if((self.T/YEAR) <= 4.0): mult = 4
        if((self.T/YEAR) <= 2.0): mult = 2
        if((self.T/YEAR) <= 1.0): mult = 1
        N = 32*mult
        if(f0 > 0.001): N = 64*mult
        if(f0 > 0.01):  N = 256*mult
        if(f0 > 0.03):  N = 512*mult
        if(f0 > 0.1):   N = 1024*mult

        M = int(math.pow(2.0,1 + int(np.log(Acut)/np.log(2.0))))
        if(M > 8192):
            M = 8192
        M = N = max(M,N)
        N *= oversample
        return(N)

    def _parse_template(self, template):
        """Return source parameters from dictionary.
        
        TODO: 
        - should be inherited from a general GB class
        - should check that keys exists
        - should also parse a vector ?
        - should get unit and check them
        """
        f0 = template["Frequency"]
        fdot = template["FrequencyDerivative"]
        theta = 0.5*np.pi-template['EclipticLatitude']
        phi = template['EclipticLongitude']
        ampl = template['Amplitude']
        incl = template['Inclination']
        psi = template['Polarization']
        phi0 = -template['InitialPhase']
        return [f0, fdot, ampl, theta, phi, psi, incl, phi0]
        
    def get_fd_tdixyz(self, template=None, f0=None, fdot=None, ampl=None,
                      theta=None, phi=None, psi=None, incl=None, phi0=None,
                      oversample=1, simulator='synthlisa'):
        """ Return TDI X,Y,Z in freq. domain. 
        
        f0 in Hz, fdot in Hz/s, ampl in strain, 
        theta,phi,psi,incl,phi0 in rad. 
        """
        if template is not None:
            [f0, fdot, ampl, theta, phi, psi, incl, phi0] = self._parse_template(template)
        pars = [f0*self.T, np.cos(theta), phi, np.log(ampl),
                np.cos(incl), psi, phi0, fdot*self.T**2]

        N = self.buffersize(f0,ampl,oversample)
        M = N  
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] xls = np.zeros(2*M)
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] xsl = np.zeros(2*M)
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] yls = np.zeros(2*M)
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] ysl = np.zeros(2*M)
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] zls = np.zeros(2*M)
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] zsl = np.zeros(2*M)
        # TODO change to complex dtype
        
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] Cpars = np.array(pars)
        
        Fast_GB(&Cpars[0], N, self.T, self.delta_t,
                &xls[0], &yls[0], &zls[0], &xsl[0], &ysl[0], &zsl[0],
                len(pars))

        lout = [xsl, ysl, zsl] if simulator=="synthlisa" else [xls, yls, zls]
        fX,fY,fZ = [np.array(a[::2] + 1.j* a[1::2], dtype=np.complex128) for a in lout]
        kmin = int(f0*self.T) - M/2
        df = 1.0/self.T
        freq = np.linspace(kmin*df,(kmin + len(fX)-1)*df, len(fX))
        self.kmin = kmin
        # TODO convert to freq. array
        return freq,fX/df,fY/df,fZ/df #return freq,fX,fY,fZ

    def get_td_tdixyz(self, **kwargs):
        """  Return TDI X,Y,Z in time domain. 
        """
        freq, fX, fY, fZ = self.get_fd_tdixyz(**kwargs)
        df = 1.0/self.T
        kmin = self.kmin#int(freq[0]/df)
        n = int(1.0/(self.delta_t*df))
        ret = np.zeros(int(n/2+1),dtype='complex128')
        ret[kmin:kmin+len(fX)] = fX*df*n; Xt = np.fft.irfft(ret)
        ret[kmin:kmin+len(fX)] = fY*df*n; Yt = np.fft.irfft(ret)
        ret[kmin:kmin+len(fX)] = fZ*df*n; Zt = np.fft.irfft(ret)

        trange = np.arange(len(Xt))*self.delta_t
        if self.T<trange[-1]:
            i_end =  np.argwhere(trange > self.T)[0][0]
        else:
            i_end = None
        return trange[:i_end], Xt[:i_end], Yt[:i_end], Zt[:i_end]
        
