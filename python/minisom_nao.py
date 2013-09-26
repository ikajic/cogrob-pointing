from minisom import MiniSom
from numpy import genfromtxt,array,linalg,zeros,mean,std,apply_along_axis
import pylab as plt
import plot_som as psom

base_path = '/home/ivana/knnl-0.1.4/knnl/build/make/data/'	

# store data in a list
data = []
#data.append(psom.fetch_data(base_path + 'hands'))
#data.append(psom.fetch_data(base_path + 'joints'))

hands = (3,4,5)
joints = (9,10,11)
		
data = [genfromtxt('/home/ivana/babbling_KB_left_arm.dat', skiprows=3, usecols=hands)[::50],
	genfromtxt('/home/ivana/babbling_KB_left_arm.dat', skiprows=3, usecols=joints)[::50]]

d = [data[0].shape[1], data[1].shape[1]]

# the size of a SOM sheet
nr_rows, nr_cols = 10, 10

# traning epochs
nr_epochs = 1000

# create networks
som = []
som.append(MiniSom(nr_rows, nr_cols, d[0], data[0], sigma=.6, learning_rate=0.5))
#som[0].random_weights_init() # choose initial nodes from data points
som[0].train_random(nr_epochs) # random training

som.append(MiniSom(nr_rows, nr_cols, d[1], data[1], sigma=.6, learning_rate=0.5))
#som[1].random_weights_init()
som[1].train_random(nr_epochs) # random training

wi_0, w_0 = som[0].get_weights()	
wi_1, w_1 = som[1].get_weights()

psom.plot_3d(final_som=w_0, data=som[0].data, init_som=wi_0, nr_nodes=nr_rows**2, title='SOM Hands')
psom.plot_3d(final_som=w_1, data=som[1].data, init_som=wi_1, nr_nodes=nr_rows**2, title='SOM Joints')

plt.show()
