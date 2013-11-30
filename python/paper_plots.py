from plot_som import *
from minisom_nao import read_data
from dbg import *

dpath = '../data/r_37min_74k.dat'

data = read_data(dpath, 100, 'u')
hands_small = fetch_som('somconf/5x5_20000_u100/')
hands_big = fetch_som('somconf/15x15_20000_u100/')

plot_3d(hands_small.get_weights()[1], data['hands'], hands_small.get_weights()[0], title='5x5 SOM over hand coordinates')

n=1
plot_3d(hands_big.get_weights()[1][::n], data['hands'], hands_big.get_weights()[0][::n], title='15x15 SOM over hand coordinates', nr_nodes=225)

show()


