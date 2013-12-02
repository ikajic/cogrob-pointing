from plot_som import *
from minisom_nao import read_data
from dbg import *

import numpy as np

dpath = '../data/r_37min_74k.dat'

data = read_data(dpath, 100, 'u')
hands_small = fetch_som('somconf/5x5_20000_u100/')
hands_big = fetch_som('somconf/15x15_20000_u100/')

#plot_3d(hands_small.weights, data['hands'], title='5x5 SOM over hand coordinates')

n=1
plot_3d(hands_big.weights[::n], data['hands'], title='')

show()

exp_data = np.genfromtxt('../experiments/nao/bigger_map_100ms/out_b100.csv', dtype=np.float, usecols=(1,4), skip_header=1, delimiter=',')

plt.figure()
time = np.linspace(0, 2.5, exp_data.shape[0])
plt.plot(time, exp_data[:,0], color='b', label='Object position')
plt.plot(time, exp_data[:,1], color='r', label='Robot\'s hand position')
plt.xlim([0, 1])
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Horizontal axis distance [mm]')

plt.legend()
