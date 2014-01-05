from plot_som import *
from minisom_nao import read_data
from dbg import *

import numpy as np

# read babbling data
dpath = '../data/r_37min_74k.dat'
spath = './plots/'
data = read_data(dpath, 100, 'u')

# read SOM configurations
n=1
hands_small = fetch_som('somconf/5x5_20000_u100/')
plot_3d(hands_small.weights[::n], data['hands'], title='')
plt.savefig(spath+'SOM_5x5.png')

hands_big = fetch_som('somconf/15x15_20000_u100/')
plot_3d(hands_big.weights[::n], data['hands'], title='')
plt.savefig(spath+'SOM_15x15.png')

show()

## read data from the experiment with human
# bigger network
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
plt.savefig(spath+'trajectory_15x15.png')

# smaller network
exp_data = np.genfromtxt('../experiments/nao/bigger_map_100ms/out_s100.csv', dtype=np.float, usecols=(1,4), skip_header=1, delimiter=',')

plt.figure()
time = np.linspace(0, 2.5, exp_data.shape[0])
plt.plot(time, exp_data[:,0], color='b', label='Object position')
plt.plot(time, exp_data[:,1], color='r', label='Robot\'s hand position')
plt.xlim([0, 1])
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Horizontal axis distance [mm]')
plt.legend()
plt.savefig(spath+'trajectory_5x5.png')
