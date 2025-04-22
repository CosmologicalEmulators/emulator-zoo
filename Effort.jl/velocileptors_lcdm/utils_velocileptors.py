import numpy as np
from velocileptors.EPT.ept_fullresum_fftw import REPT
from classy import Class

# For the wiggle no-wiggle split of Pk
#from scipy.special import hyp2f1
#from scipy.interpolate import interp1d
#from scipy.ndimage import gaussian_filter
#from scipy.signal import argrelmin, argrelmax
#from scipy.fftpack import dst, idst
#from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import scipy

def pnw_dst(k,p, ii_l=None,ii_r=None,extrap_min=1e-3, extrap_max=10, N=16):
    
    '''
    Implement the wiggle/no-wiggle split procedure from Benjamin Wallisch's thesis (arXiv:1810.02800)
    
    '''

    # put onto a linear grid
    ks = np.linspace( extrap_min, extrap_max, 2**N)
    lnps = scipy.interpolate.InterpolatedUnivariateSpline(k, np.log(k*p), ext=1)(ks)
 
    
    # sine transform
    dst_ps = scipy.fftpack.dst(lnps)
    dst_odd = dst_ps[1::2]
    dst_even = dst_ps[0::2]
    
    # find the BAO regions
    if ii_l is None or ii_r is None:
        d2_even = np.gradient( np.gradient(dst_even) )
        ii_l = scipy.signal.argrelmin(scipy.ndimage.gaussian_filter(d2_even,4))[0][0]
        ii_r = scipy.signal.argrelmax(scipy.ndimage.gaussian_filter(d2_even,4))[0][1]
        #print(ii_l,ii_r)
    
        iis = np.arange(len(dst_odd))
        iis_div = np.copy(iis); iis_div[0] = 1.
        #cutiis_odd = (iis > (ii_l-3) ) * (iis < (ii_r+20) )
        cutiis_even = (iis > (ii_l-3) ) *  (iis < (ii_r+10) )
        
        d2_odd = np.gradient( np.gradient(dst_odd) )
        ii_l = scipy.signal.argrelmin(scipy.ndimage.gaussian_filter(d2_odd,4))[0][0]
        ii_r = scipy.signal.argrelmax(scipy.ndimage.gaussian_filter(d2_odd,4))[0][1]
        #print(ii_l,ii_r)
    
        iis = np.arange(len(dst_odd))
        iis_div = np.copy(iis); iis_div[0] = 1.
        cutiis_odd = (iis > (ii_l-3) ) * (iis < (ii_r+20) )
        #cutiis_even = (iis > (ii_l-3) ) *  (iis < (ii_r+10) )
        
    else:
        iis = np.arange(len(dst_odd))
        iis_div = np.copy(iis); iis_div[0] = 1.
        cutiis_odd = (iis > (ii_l) ) * (iis < (ii_r) )
        cutiis_even = (iis > (ii_l) ) *  (iis < (ii_r) )

    # ... and interpolate over them
    interp_odd = scipy.interpolate.interp1d(iis[~cutiis_odd],(iis**2*dst_odd)[~cutiis_odd],kind='cubic')(iis)/iis_div**2 
    interp_odd[0] = dst_odd[0]
    
    interp_even = scipy.interpolate.interp1d(iis[~cutiis_even],(iis**2*dst_even)[~cutiis_even],kind='cubic')(iis)/iis_div**2 
    interp_even[0] = dst_even[0]
    
    # Transform back
    interp = np.zeros_like(dst_ps)
    interp[0::2] = interp_even
    interp[1::2] = interp_odd

    lnps_nw = scipy.fftpack.idst(interp) / 2**17
    
    return k, scipy.interpolate.InterpolatedUnivariateSpline(ks, np.exp(lnps_nw)/ks,ext=1)(k)
def D_of_a(a,OmegaM=1):
    # From Stephen Chen
    return a * scipy.special.hyp2f1(1./3,1,11./6,-a**3/OmegaM*(1-OmegaM)) / scipy.special.hyp2f1(1./3,1,11./6,-1/OmegaM*(1-OmegaM))

def f_of_a(a, OmegaM=1):
    # From Stephen Chen
    Da = D_of_a(a,OmegaM=OmegaM)
    ret = Da/a - a*(6*a**2 * (1 - OmegaM) * scipy.special.hyp2f1(4./3, 2, 17./6, -a**3 *  (1 - OmegaM)/OmegaM))/(11*OmegaM)/scipy.special.hyp2f1(1./3,1,11./6,-1/OmegaM*(1-OmegaM))
    return a * ret / Da
