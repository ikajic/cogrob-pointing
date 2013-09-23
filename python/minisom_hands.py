from minisom import MiniSom
from plot_som import fetch_data, plot_data3d
from numpy import genfromtxt,array,linalg,zeros,mean,std,apply_along_axis
import pylab as plt
from copy import copy

base_path = '/home/ivana/knnl-0.1.4/knnl/build/make/data/'	

# store data in a list
data = []
data.append(fetch_data(base_path + 'hands'))
data.append(fetch_data(base_path + 'joints'))

data[0] = apply_along_axis(lambda x: x/linalg.norm(x),1, data[0]) # data normalization
data[1] = apply_along_axis(lambda x: x/linalg.norm(x),1, data[1]) # data normalization

d = [data[0].shape[1], data[1].shape[1]]

# network and training parameters 
nr_rows, nr_cols = 10, 10
nr_epochs = 1000

# create networks
som = []
som.append(MiniSom(nr_rows, nr_cols, d[0], sigma=.6, learning_rate=0.5))
som[0].random_weights_init(data[0])
som[0].train_random(data[0], nr_epochs) # random training

som.append(MiniSom(nr_rows, nr_cols, d[1], sigma=.6, learning_rate=0.5))
som[1].random_weights_init(data[1])
winit = copy(som[1].weights).reshape(nr_rows*nr_cols, d[1])
som[1].train_random(data[1], nr_epochs) # random training

plot_data3d(som[0].weights.reshape(nr_cols*nr_rows, d), data[0][::20, :], nr_nodes=100, title='Hands SOM')
plot_data3d(som[1].weights.reshape(nr_cols*nr_rows, d), data[1][::20, :], winit, nr_nodes=100, title='Joints SOM')

plt.show()
