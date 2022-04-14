#!/usr/bin/env python

"""
BBH:
python GWFish/CBC_Background.py --pop_file GWFish/injections/BBH_1e5.hdf5 --config GWFish/GWFish/detectors.yaml --outdir /home/boris.goncharov/null_stream_out/gwfish/

BNS:
python GWFish/CBC_Background.py --pop_file GWFish/injections/BNS_pop.hdf5 --config GWFish/GWFish/detectors.yaml --outdir /home/boris.goncharov/null_stream_out/gwfish/
"""

import numpy as np
import pandas as pd

from numpy.random import default_rng

import time
import json
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  #"font.serif": ["Palatino"],
})
font = {'family' : 'serif',
        'size'   : 17}

from astropy.cosmology import Planck13
from astropy import units as u

import pickle
import os
import argparse

import GWFish.modules as gw

h0_si = Planck13.H0.to(1/u.s).value
def psd_stochastic(ff, f_ref, omega_ref,alpha=2./3.):
    return omega_ref*(ff/f_ref)**alpha * 3 * h0_si**2/2/np.pi**2*ff**(-3)

rng = default_rng()

def analyzeForeground(network, h_of_f, dT, plotname, pop_name):
    for d in np.arange(len(network.detectors)):
        ff = network.detectors[d].frequencyvector[:,0]

        psd_stoch = {
          'BBH': {
            'high': psd_stochastic(ff,25,6.3*10**(-10)),
            'low': psd_stochastic(ff,25,3.4*10**(-10)),
            'max': psd_stochastic(ff,25,4.7*10**(-10))
          },
          'BNS': {
            'high': psd_stochastic(ff,25,5.2*10**(-10)),
            'low': psd_stochastic(ff,25,0.6*10**(-10)),
            'max': psd_stochastic(ff,25,2.0*10**(-10))
          }
        }
        components = network.detectors[d].components
        psd_astro_all = np.abs(np.squeeze(h_of_f[:, d, :])) ** 2 / dT
        N = len(psd_astro_all[0, :])

        #plotrange = components[0].plotrange

        # Not a good physical representation
        #for realization in psd_astro_all.T:
        #  plt.loglog(ff, realization, alpha=1/psd_astro_all.shape[1]**0.25, color='brown', linewidth=0.1)

        bb = np.logspace(-54, -43, 1000)
        aa = np.logspace(np.log10(2),np.log10(1000),1000)
        hist = np.zeros((len(bb) - 1, len(aa)))
        for aa_idx, aa_val in enumerate(aa):
            #idx_ff = (np.abs(ff - aa_val)).argmin()
            hist[:, aa_idx] = np.histogram(psd_astro_all[(np.abs(ff - aa_val)).argmin(), :], bins=bb)[0]
        bb = np.delete(bb, -1)

        # calculate percentiles
        hist_norm = hist / N
        hist_norm[np.isnan(hist_norm)] = 0
        histsum = np.cumsum(hist_norm, axis=0) / (np.sum(hist_norm, axis=0)[np.newaxis, :])
        ii10 = np.argmin(np.abs(histsum - 0.1), axis=0)
        ii50 = np.argmin(np.abs(histsum - 0.5), axis=0)
        ii90 = np.argmin(np.abs(histsum - 0.9), axis=0)

        hist[hist == 0] = 0#0.0001#np.nan

        #from scipy.interpolate import interp2d
        #bb_fine = np.logspace(-26, -21.5, 1000)
        #aa_fine = np.logspace(np.log10(2),np.log10(1000),1000)
        #interp_hist = interp2d(aa, bb, hist, kind='cubic')
        #hist = interp_hist(aa_fine, bb_fine)
        #aa = aa_fine
        #bb = bb_fine

        fig = plt.figure()#figsize=(9, 6))
        axes = fig.add_subplot(111)
        axes.set_facecolor('#FFFAEB')
        #cmap = plt.get_cmap('RdYlBu_r')
        cmap = plt.get_cmap('gist_earth')#cubehelix')
        #cm = plt.pcolormesh(np.transpose(aa), bb, hist, cmap=cmap, norm=LogNorm(vmin=0.1, vmax=np.max(hist)),linewidth=0,rasterized=True) # For 0 = 0.0001
        cm = plt.pcolormesh(np.transpose(aa), bb, hist, cmap=cmap, norm=LogNorm(vmin=1, vmax=np.max(hist)),linewidth=0,rasterized=True) # For 0 = nan
        # plt.loglog(ff, h_astro)
        plt.loglog(aa, components[0].Sn(aa), color='#D10000', alpha = 0.9, linewidth = 1.5)
        #plt.loglog(aa, np.sqrt(components[0].Sn(aa)), color='#FFFFFF', alpha = 0.8, linewidth = 1.2, linestyle='--')
        plt.fill_between(ff,psd_stoch[pop_name]['low'],psd_stoch[pop_name]['high'],color='#F87217',alpha=0.6,label='BBH')
        #plt.loglog(ff,np.sqrt(psd_stoch[pop_name]['max']),color='red',linewidth=0.5,alpha=0.9)
        plt.loglog(aa, bb[ii10], 'w-',linewidth=0.4)
        plt.loglog(aa, bb[ii50], 'w-',linewidth=0.4)
        plt.loglog(aa, bb[ii90], 'w-',linewidth=0.4)
        #plt.loglog(aa, bb[ii10], 'k--',linewidth=0.7)
        #plt.loglog(aa, bb[ii50], 'k--',linewidth=0.7)
        #plt.loglog(aa, bb[ii90], 'k--',linewidth=0.7)
        axes.set_xlabel('$\mathrm{Frequency~[Hz]}$', fontdict=font)
        #axes.set_ylabel('$\mathrm{Strain~spectra~[Hz}^{-1/2}\mathrm{]}$', fontdict=font)
        axes.set_ylabel('$\mathrm{PSD~[s]}$', fontdict=font)
        axes.tick_params(axis='y', labelsize = font['size'])
        axes.tick_params(axis='x', labelsize = font['size'])
        #plt.xlabel('Frequency [Hz]', fontsize=20)
        #plt.ylabel(r"Strain spectra [$1/\sqrt{\rm Hz}$]", fontsize=20)
        plt.xlim([2,1000])
        plt.ylim([10**(-52),10**(-43.6)])
        cb = plt.colorbar()
        cb.set_label(label='$N$',size=font['size'],family=font['family'])
        cb.ax.tick_params(labelsize=font['size'])
        #plt.grid(True)
        #fig.colorbar(cm)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
        #plt.savefig('Astrophysical_histo_' + components[0].name + '.png', dpi=300)
        plt.savefig(plotname+'.png')
        plt.savefig(plotname+'.pdf')
        plt.close()

        # THIS WORKS OK:
        #plt.figure()
        #plt.loglog(ff, np.sqrt(psd_astro_all[:,0]))
        #plt.loglog(ff, np.sqrt(components[0].Sn(ff)), color='green')
        #plt.savefig(plotname+'_one.png')
        #plt.close()

        #from matplotlib.colors import LogNorm
        #plt.imshow(hist, aspect='auto',origin='lower',extent=(np.min(aa),np.max(aa),np.min(bb),np.max(bb)),norm=LogNorm(vmin=0.1, vmax=np.max(hist)))
        #plt.plot(ff, np.sqrt(components[0].Sn(ff)), color='green')
        #plt.savefig(plotname+'.imshow.png')
        #plt.close()

def main():
    # example to run with command-line arguments:
    # python CBC_Foreground.py --pop_file=CBC_pop.hdf5 --detectors ET CE2

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop_file', type=str, default='./injections/BBH_1e5.hdf5', nargs=1,
        help='Population to run the analysis on.'
             'Runs on BBH_1e5.hdf5 if no argument given.')
    parser.add_argument(
        '--detectors', type=str, default=['ET'], nargs='+',
        help='Detectors to analyze. Uses ET as default if no argument given.')
    parser.add_argument(
        '--outdir', type=str, default='./', 
        help='Output directory.')
    parser.add_argument(
        '--config', type=str, default='GWFish/detectors.yaml',
        help='Configuration file where the detector specifications are stored. Uses GWFish/detectors.yaml as default if no argument given.')

    args = parser.parse_args()
    ConfigDet = args.config

    #popname = 'BBH'
    #dT = 60
    #N = 7200
    #dT = 24*3600
    #N = 300
    #dT = 3600
    #N = 1000

    popname = 'BNS'
    #dT = 60
    #N = 7200
    dT = 24*3600
    N = 300

    t0 = 1104105616

    threshold_SNR = 5000  # min. network SNR for detection
    duty_cycle = False  # whether to consider the duty cycle of detectors

    pop_file = args.pop_file

    detectors_ids = args.detectors
    parameters = pd.read_hdf(pop_file[0])

    ns = len(parameters)

    network = gw.detection.Network(detectors_ids, detection_SNR=threshold_SNR, parameters=parameters,
                                   fisher_parameters=None, config=ConfigDet)

    waveform_model = 'lalsim_IMRPhenomD'
    #waveform_model = 'gwfish_TaylorF2'
    # waveform_model = 'lalbbh_TaylorF2'

    frequencyvector = network.detectors[0].frequencyvector

    h_of_f = np.zeros((len(frequencyvector), len(network.detectors), N), dtype=complex)
    cnt = np.zeros((N,))

    fname = '_'.join([popname,str(dT),str(N)])

    if os.path.exists(args.outdir+fname+'.txt'):
        print('Loading CBC population')
        h_of_f = np.loadtxt(args.outdir+fname+'.txt').view(complex)
        h_of_f = np.expand_dims(h_of_f, 1)
        print('Loading completed')
    else:
        print('Processing CBC population')
        for k in tqdm(np.arange(ns)):
            parameter_values = parameters.iloc[k]
            tc = parameter_values['geocent_time']
    
            # make a precut on the signals; note that this depends on how long signals stay in band (here not more than 3 days)
            if ((tc>t0) & (tc-3*86400<t0+N*dT)):
                signals = np.zeros((len(frequencyvector), len(network.detectors)), dtype=complex)  # contains only 1 of 3 streams in case of ET
                for d in np.arange(len(network.detectors)):
                    wave, t_of_f = gw.waveforms.hphc_amplitudes(waveform_model, parameter_values,
                                                                network.detectors[d].frequencyvector)
    
                    det_signals = gw.detection.projection(parameter_values, network.detectors[d], wave, t_of_f)
                    signals[:,d] = det_signals[:,0]
    
                    SNRs = gw.detection.SNR(network.detectors[d], det_signals, duty_cycle=duty_cycle)
                    network.detectors[d].SNR = np.sqrt(np.sum(SNRs ** 2))
    
                SNRsq = 0
                for detector in network.detectors:
                    SNRsq += detector.SNR ** 2
    
                if (np.sqrt(SNRsq) < threshold_SNR):
                    for n in np.arange(N):
                        t1 = t0+n*dT
                        t2 = t1+dT
                        ii = np.argwhere((t_of_f[:,0] < t1) | (t_of_f[:,0] > t2))
                        signals_ii = np.copy(signals)
    
                        if (len(ii) < len(t_of_f)):
                            #print("Signal {0} contributes to segment {1}.".format(k,n))
                            cnt[n] += 1
                            signals_ii[ii,:] = 0
                            h_of_f[:,:,n] += signals_ii

        np.savetxt(args.outdir+fname+'.txt',np.squeeze(h_of_f).view(float))

    analyzeForeground(network, h_of_f, dT, args.outdir+fname,popname)

    print('Out of {0} signals, {1} are in average undetected binaries falling in a {2}s time window.'.format(ns, np.mean(cnt), dT))

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
