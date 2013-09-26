from mpl_toolkits.mplot3d import Axes3D
import pylab as plt
import numpy as np
import pdb

def fetch_data(path, skip = 0):
	'''
	Columnwise stored data used by SOMs (either neural weights or input data).
	''' 
	data = np.genfromtxt(path, dtype='float', skiprows=skip)
	return data

def plot_weights(init_som, final_som, title=['SOM init', 'SOM final'], dim_lab=None):
	'''
	Function to plot neural weights before and after the training for each dimension.
	'''	
	
	assert init_som.shape == final_som.shape
	
	n, d = init_som.shape
	width = np.int(np.sqrt(n))
	
	if dim_lab is None:
		dim_lab = ['w' + str(i) for i in xrange(d)]

	fig = plt.figure()
	
	for lab, i in zip(dim_lab, xrange(d)):
			
		# plot weights before training
		plt.suptitle(title[0], fontsize = 14)
		ax = fig.add_subplot(2, d, i+1)
		img = init_som[:, i].reshape(width, width)
		ax.imshow(img, interpolation='nearest')
		plt.title(lab)

		# same weights after training

		ax = fig.add_subplot(2, d, (i+1) + d)
		if i==int(d/2.): plt.title(title[1])
		img_f = final_som[:, i].reshape(width, width)
		ax.imshow(img_f, interpolation='nearest')

def plot_3d(final_som, data, init_som=None, title=None, nr_nodes=50):
	'''
	3D plot of input data, initial positions of neurons and positions of neurons after training.
	If data have more than 3 dimensions, plot only the first three.
	'''

	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	
	if data.shape[1] > 3:
		print "using only 3 dimensions for plotting"
		data = data[:,:3]

	if init_som is not None:
		ax.plot(init_som[:nr_nodes,0], init_som[:nr_nodes,1], init_som[:nr_nodes,2], c='g', marker='o', label='init', linestyle='None', alpha=0.6, markersize=3)
		
	d = ax.plot(data[:, 0], data[:,1], data[:,2], c='b', marker='*', linestyle='None', alpha=0.4, label='data')
	n = ax.plot(final_som[:nr_nodes,0], final_som[:nr_nodes,1], final_som[:nr_nodes,2], c='r', marker='o', alpha = 0.6, label='neurons', markersize=4)
	
	ax.legend(numpoints=1)
	fig.suptitle(title)

def show():
	plt.show()

