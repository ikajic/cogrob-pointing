from mpl_toolkits.mplot3d import Axes3D
import pylab as plt
import numpy as np
import pdb

def fetch_data(path, skip = 0):
	'''
	Columnwise stored data used by SOMs (either neural weights or input data).
	''' 
	with open(path) as file:	
		for _ in xrange(skip):
			next(file)
		rows = [filter(None, [x.strip() for x in line.split('\t')]) for line in file]
	data = np.array(rows).astype(float)
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
		ax.imshow(init_som[:, i].reshape(width, width), interpolation='nearest')
		plt.title(lab)

		# same weights after training

		ax = fig.add_subplot(2, d, (i+1) + d)
		if i==int(d/2.): plt.title(title[1])
		ax.imshow(final_som[:, i].reshape(width, width), interpolation='nearest')

def plot_data3d(init_som, final_som, data, title=None, nr_nodes=50):
	'''
	3D plot of input data, initial positions of neurons and positions of neurons after training
	'''

	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.scatter(init_som[:nr_nodes,0], init_som[:nr_nodes,1], init_som[:nr_nodes,2], c='g', marker='o')
	ax.plot(final_som[:nr_nodes,0], final_som[:nr_nodes,1], final_som[:nr_nodes,2], c='r', marker='o', alpha = 0.4)
	ax.scatter(data[:, 0], data[:,1], data[:,2], c='b', marker='.', alpha=0.2)

	#TODO: add legend somehow

