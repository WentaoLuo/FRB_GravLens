#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
nx   = 80
ny   = 400
freq = np.linspace(800,400,ny)
dtim0= np.linspace(0,40,nx)
dtim1= np.linspace(-40,0,nx)
fpeak= 1.7         # peak flux in Jy
sigt = 2.0
sign = 0.2
xshft= 0.0
Dm   = 500.0       # 
pi   = np.pi
c    = 299792458.0 # speed of light m/s
elct = 1.602176634e-19 # electron charge in Columbus
me   = 9.10938356e-31  # electron mass in kg
fact = elct*elct/2.0/pi/me/c
fctr = fact*Dm*1.0280736*1e+20
patch= np.zeros(nx)
def frb1d():
  dtim  = dtim0
  flx   = fpeak*(dtim/sigt/sigt)*np.exp(-0.5*(dtim*dtim/sigt/sigt))
  noise1= np.random.normal(loc=0.0,scale=sign,size=nx)
  noise2= np.random.normal(loc=0.0,scale=sign,size=nx)
  flux  = flx+noise1
  total = np.concatenate((noise2,flux))
  ffx   = np.concatenate((patch,flx))
  dtot  = np.concatenate((dtim1,dtim0))
  return [dtot,total,ffx] 
def waterfall(freq_center,bandwidth):
  tx,fx,ffx=frb1d()
  image= np.zeros((2*nx,ny))
  for i in range(ny-1):
    noise= np.random.normal(loc=0.0,scale=sign,size=2*nx)
    if freq[i] <=freq_center:
       dfreq2inv = (1.0/freq[i+1]/freq[i+1]-1.0/freq[i]/freq[i])
       tinterval = 1000.0*fctr*dfreq2inv
       image[:,i]= shift(ffx,tinterval)+noise
    if freq[i]<=freq_center-bandwidth:
       image[:,i]= noise
    if freq[i]>freq_center:
       image[:,i]= noise
  plt.subplot(2,1,1)
  plt.plot(tx,ffx+noise,'k-',lw=3.0)
  plt.ylabel(r'$\mathrm{Flux(Jy)}$',fontsize=15.0)
  plt.xticks(())
  plt.yticks((0.0,0.5,0.8))
  plt.ylim(-0.2,0.8)
  plt.subplot(2,1,2)
  plt.imshow(image.T,aspect='auto',interpolation='nearest',extent=[-80,80,400,800])
  plt.xlim(-80,80)
  plt.ylim(400,800)
  plt.yticks((400.0,600,800))
  plt.xlabel(r'$\mathrm{\Delta t(ms)}$',fontsize=15.0)
  plt.ylabel(r'$\mathrm{Frequency(MHz)}$',fontsize=15.0)
  #plt.colorbar()
  plt.subplots_adjust(hspace=0.00)
  plt.show()
  return 0
def main():
  freq_center = 700.0  # FRB freq started to be observed
  bandwidth   = 200.0
  #tx,fy=frb1d()
  #plt.plot(tx,fy,'r-')
  #plt.show()
  waterfall(freq_center,bandwidth)

if __name__=='__main__':
  main()
