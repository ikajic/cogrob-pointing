from minisom import MiniSom
from plot_som import fetch_data, plot_data3d
from numpy import genfromtxt,array,linalg,zeros,mean,std,apply_along_axis
import pylab as plt

base_path = '/home/ivana/knnl-0.1.4/knnl/build/make/data/'	

data = fetch_data(base_path + 'hands')
data = apply_along_axis(lambda x: x/linalg.norm(x),1,data) # data normalization

d = data.shape[1]
nr_rows, nr_cols = 20, 20
nr_epochs = 1000


som = MiniSom(nr_rows, nr_cols, d, sigma=.7, learning_rate=0.1)
som.random_weights_init(data)
w_init = som.weights
som.train_random(data, nr_epochs) # random training

plot_data3d( som.weights.reshape(nr_cols*nr_rows, d), data[::50, :], init_som=w_init)
plt.show()
