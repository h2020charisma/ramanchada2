import numpy as np
import ramanchada2 as rc2
import matplotlib.pyplot as plt
import os

def test_pearson4():
    x1 = np.loadtxt('./tests/data/experimental/x_axis.txt')
    y1 = np.loadtxt('./tests/data/experimental/y_axis.txt')
    spe0 = rc2.spectrum.Spectrum(x=x1, y=y1)
    peak_candidates = spe0.find_peak_multipeak(sharpening= None, strategy='topo', hht_chain = [150], wlen = 200)
    fitres = spe0.fit_peak_multimodel(profile='Pearson4', candidates=peak_candidates, no_fit=False)
    fig, ax = plt.subplots(figsize=(35, 10))
    fitres.plot(ax=ax,label='peak fitted')
    spe0.plot(ax=ax,label='experimental')
    plt.savefig('pearson4.png')

def test_generate_and_fit_clean():
    lines = {40: 20, 150: 15, 200: 30, 500: 50, 550: 5}
    spe = rc2.spectrum.from_delta_lines(lines).convolve('pearson4', sigma=3)
    dofit(spe,lines,shift=0)

def test_generate_and_fit_noise():
    lines = {40: 20, 150: 15, 200: 30, 500: 50, 550: 5}
    spe = rc2.spectrum.from_delta_lines(lines).convolve('pearson4', sigma=3)
    spe = spe.add_baseline(n_freq=5, amplitude=3, pedestal=0, rng_seed=1111)
    spe = spe.add_gaussian_noise(.1, rng_seed=1111)
    #spe = spe.scale_xaxis_fun(lambda x: x - shift)
    dofit(spe,lines,shift=0,file_fig='pearson4_synthetic_noise.png')    

def test_generate_and_fit_noise_shift():
    lines = {40: 20, 150: 15, 200: 30, 500: 50, 550: 5}
    shift = 50
    spe = rc2.spectrum.from_delta_lines(lines).convolve('pearson4', sigma=3)
    spe = spe.add_baseline(n_freq=5, amplitude=3, pedestal=0, rng_seed=1111)
    spe = spe.add_gaussian_noise(.1, rng_seed=1111)
    spe = spe.scale_xaxis_fun(lambda x: x - shift)
    dofit(spe,lines,shift=shift,file_fig='pearson4_synthetic_noise_shift.png')       

def dofit(spe,lines,shift=0,file_fig='pearson4_synthetic.png'):    
    candidates = spe.find_peak_multipeak(prominence=spe.y_noise*5, wlen=40, sharpening=None)

    true_pos = np.array(list(lines.keys()))
    calc_pos = [i for gr in candidates for i in gr.positions]
    fit_peaks = spe.fit_peak_multimodel(profile='Pearson4', candidates=candidates)
    print(fit_peaks.to_dataframe_peaks())
    fit_pos = fit_peaks.locations

    fig, ax = plt.subplots(figsize=(24, 8))
    fit_peaks.plot(ax=ax,individual_peaks=True)    
    spe.plot(ax=ax,label='synthetic')
    plt.savefig(file_fig)    
    print(fit_peaks)
    assert len(true_pos) == len(calc_pos), 'wrong number of peaks found'
    assert np.max(np.abs(true_pos - fit_pos - shift)) < 3, 'fit locations far from truth'
    assert np.max(np.abs(true_pos - calc_pos - shift)) < 5, 'find_peaks locations far from truth'

