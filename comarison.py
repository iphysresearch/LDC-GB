
def construct_slow_part(T, arm_length, Ps, tm, f0, fdot, fstar, phi0, k, DP, DC, eplus, ecross, N=512):
    P1, P2, P3 = Ps
    r = dict()
    r['12'] = (P2 - P1)/arm_length ## [3xNt]
    r['13'] = (P3 - P1)/arm_length
    r['23'] = (P3 - P2)/arm_length
    r['31'] = -r['13']

    kdotr = dict()
    for ij in ['12', '13', '23']:
        kdotr[ij] = np.dot(k, r[ij]) ### should be size Nt
        kdotr[ij[-1]+ij[0]] = -kdotr[ij]

    kdotP = np.array([np.dot(k, P1),
                      np.dot(k, P2),
                      np.dot(k, P3)])
    kdotP /= CLIGHT
    
    Nt = len(tm)
    xi = tm - kdotP
    fi = f0 + fdot*xi
    fonfs = fi/fstar #Ratio of true frequency to transfer frequency

    ### compute transfer f-n
    q = np.rint(f0 * T) # index of nearest Fourier bin
    df = 2.*np.pi*(q/T)
    om = 2.0*np.pi*f0
    ### The expressions below are arg2_i with om*kR_i factored out

    A = dict()
    for ij in ['12', '23', '31']:
        aij = np.dot(eplus,r[ij])*r[ij]*DP+np.dot(ecross,r[ij])*r[ij]*DC
        A[ij] = aij.sum(axis=0)
    # These are wfm->TR + 1j*TI in c-code

    # arg2_1 = 2.0*np.pi*f0*xi[0] + phi0 - df*tm + np.pi*fdot*(xi[0]**2)
    # arg2_2 = 2.0*np.pi*f0*xi[1] + phi0 - df*tm + np.pi*fdot*(xi[1]**2)
    # arg2_3 = 2.0*np.pi*f0*xi[2] + phi0 - df*tm + np.pi*fdot*(xi[2]**2)

    ### These (y_sr) reproduce exactly the FastGB results 
    #self.y12 = 0.25*np.sin(arg12)/arg12 * np.exp(1.j*(arg12 + arg2_1)) * ( Dp12*self.DP + Dc12*self.DC )
    #self.y23 = 0.25*np.sin(arg23)/arg23 * np.exp(1.j*(arg23 + arg2_2)) * ( Dp23*self.DP + Dc23*self.DC )
    #self.y31 = 0.25*np.sin(arg31)/arg31 * np.exp(1.j*(arg31 + arg2_3)) * ( Dp31*self.DP + Dc31*self.DC )
    #self.y21 = 0.25*np.sin(arg21)/arg21 * np.exp(1.j*(arg21 + arg2_2)) * ( Dp12*self.DP + Dc12*self.DC )
    #self.y32 = 0.25*np.sin(arg32)/arg32 * np.exp(1.j*(arg32 + arg2_3)) * ( Dp23*self.DP + Dc23*self.DC )
    #self.y13 = 0.25*np.sin(arg13)/arg13 * np.exp(1.j*(arg13 + arg2_1)) * ( Dp31*self.DP + Dc31*self.DC )

    ### Those are corrected values which match the time domain results.
    ## om*kdotP_i singed out for comparison with another code.
    argS =  phi0 + (om - df)*tm + np.pi*fdot*(xi**2)
    kdotP = om*kdotP - argS
    Gs = dict()
    for ij, ij_sym, s in [('12', '12', 0), ('23', '23', 1), ('31', '31', 2),
                          ('21', '12', 1), ('32', '23', 2), ('13', '31', 0)]:
        arg_ij = 0.5*fonfs[s,:] * (1 + kdotr[ij])
        Gs[ij] = 0.25*np.sinc(arg_ij/np.pi) * np.exp(-1.j*(arg_ij + kdotP[s])) * A[ij_sym]
        
    ### Lines blow are extractions from another python code and from C-code
    # y = -0.5j*self.omL*A*sinc(args)*np.exp(-1.0j*(args + self.om*kq))
    # args = 0.5*self.omL*(1.0 - kn)
    # arg12 = 0.5*fonfs[0,:] * (1 + kdotr12)
    # arg2_1 = 2.0*np.pi*f0*xi[0] + phi0 - df*tm + np.pi*self.fdot*(xi[0]**2)  -> om*k.Ri
    # arg1 = 0.5*wfm->fonfs[i]*(1. + wfm->kdotr[i][j]);
    # arg2 =  PI*2*f0*wfm->xi[i] + phi0 - df*t;
    # sinc = 0.25*sin(arg1)/arg1;
    # tran1r = aevol*(wfm->dplus[i][j]*wfm->DPr + wfm->dcross[i][j]*wfm->DCr);
    # tran1i = aevol*(wfm->dplus[i][j]*wfm->DPi + wfm->dcross[i][j]*wfm->DCi);
    # tran2r = cos(arg1 + arg2);
    # tran2i = sin(arg1 + arg2);
    # wfm->TR[i][j] = sinc*(tran1r*tran2r - tran1i*tran2i);
    # wfm->TI[i][j] = sinc*(tran1r*tran2i + tran1i*tran2r);
    return Gs, q


