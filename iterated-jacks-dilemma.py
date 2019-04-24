import os
import numpy as np
import matplotlib.pyplot as plt


data = np.load('geo.npz')
print(data.keys())
strata = data['strata']
density = data['density']
horizon = np.argmax(strata, axis=0)
halfw = 60
pln_index_raw = (horizon +
                 np.arange(-halfw, halfw)[:, np.newaxis, np.newaxis])
pln_index = pln_index_raw % density.shape[0]
row_index = np.arange(512)[np.newaxis, :, np.newaxis]
col_index = np.arange(512)[np.newaxis, np.newaxis, :]
shifted = density[pln_index, row_index, col_index].astype(float)
shifted[pln_index_raw < 0] = np.nan
shifted[pln_index_raw >= density.shape[0]] = np.nan
fig, ax = plt.subplots()
ax.plot(np.nanmean(shifted, axis=(1, 2)))
ax.axvline(x=halfw, c='r')

plt.show(block=True)
