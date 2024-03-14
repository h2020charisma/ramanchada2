import numpy as np
import ramanchada2 as rc2
import matplotlib.pyplot as plt
import os

"""
TODO To include proper tests in future!
"""
x1 = np.loadtxt('./tests/peak/x_axis.txt')
y1 = np.loadtxt('./tests/peak/y_axis.txt')
spe0 = rc2.spectrum.Spectrum(x=x1, y=y1)

peak_candidates = spe0.find_peak_multipeak(sharpening= None, strategy='topo', hht_chain = [150], wlen = 200)

fitres = spe0.fit_peak_multimodel(profile='Pearson4', candidates=peak_candidates, no_fit=False)

fig, ax = plt.subplots(figsize=(35, 10))

fitres.plot(ax=ax)
spe0.plot(ax=ax)


plt.show()