from __future__ import division

from numpy import genfromtxt, zeros, product, setdiff1d, arange, where, set_printoptions, unravel_index, tanh, savetxt
from parameters import param
from random import choice

import plot_som as ps
import random
import sys
import os
import pdb

from minisom import MiniSom
from similar_vec import get_similar_data

#TODO
import pylab as plt

def get_path():
	"""
	Path to .dat file generated by NAO's babbling is given by a user in 
	terminal. If valid, return. 
	"""
	nrarg = len(sys.argv)

	if nrarg<2:
		raise Exception('Missing data path')

	path = str(sys.argv[1])

	if not os.path.exists(path):
		raise Exception('Path doesn\'t exist')

	return path

def read_data(path, nrpts=50):
	"""
	Return babbling coordinates for hands and joints.
		nrpts - take each nrpts-th coordinate for training
	"""
	
	hands = param['hands']
	joints = param['joints']
		
	data = {
		'hands': genfromtxt(path, skiprows=3, usecols=hands)[:nrpts],
		'joints': genfromtxt(path, skiprows=3, usecols=joints)[:nrpts]
		}

	return data 

def train_som(data):
	
	som = MiniSom(
		param['nr_rows'],
		param['nr_cols'], 
		data.shape[1], 
		data, 
		sigma=param['sigma'], 
		learning_rate=param['learning_rate'], 
		norm='minmax')
		
	#som.random_weights_init() # choose initial nodes from data points
	som.train_random(param['nr_epochs']) # random training
	
	return som

def hebbian_learning(som1, som2):
	s1, s2 = som1.weights.shape, som2.weights.shape
	hebb = zeros((param['nr_rows'], param['nr_cols'], \
		param['nr_rows'], param['nr_cols']))
	
	f = lambda x: 1/(1+tanh(x))
	for dp1, dp2 in zip(som1.data, som2.data):
		#pdb.set_trace()
		act1 = som1.activate(dp1)
		act2 = som2.activate(dp2)
				
		idx1 = som1.winner(dp1)
		idx2 = som2.winner(dp2)
		
		hebb[idx1[0], idx1[1], idx2[0], idx2[1]] += param['eta'] * f(act1[idx1]) * f(act2[idx2])
		
	return hebb


# useful plotting, TODO extract to plot som
def plot(som_hands, som_joints):
	wi_0, w_0 = som_hands.get_weights()	
	wi_1, w_1 = som_joints.get_weights()

	ps.plot_3d(final_som=w_0, data=som_hands.data, init_som=wi_0, nr_nodes=param['n_to_plot'], title='SOM Hands')
	ps.plot_3d(final_som=w_1, data=som_joints.data, init_som=wi_1, nr_nodes=param['n_to_plot'], title='SOM Joints')


def plot_inactivated_nodes(som, inact):
	_, w = som.get_weights()	
	act = setdiff1d(arange(w.shape[0]), inact)
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	data = som.data
	
	# Just a fraction of data used for plotting
	data = data[:,:3][::20]
	d = ax.plot(data[:, 0], data[:,1], data[:,2], c='b', marker='*', linestyle='None', alpha=0.4, label='data')
	
	# Plot activated nodes in green
	act_nod = w[act, :]
	a = ax.plot(act_nod[:, 0], act_nod[:, 1], act_nod[:, 2], c='g', marker='o', alpha = 0.6, label='neurons', markersize=4)
	
	# Plot inactivated nodes in red
	in_w = w[inact, :]
	i = ax.plot(in_w[:, 0], in_w[:, 1], in_w[:, 2], c='r', marker='o', alpha = 0.6, label='inact. neurons', markersize=6, linestyle='None')
	plt.legend(numpoints=1)
	
def print_strongest_connections(hebb_weights):
	# print the strongest connections
	for i in xrange(param['nr_rows']):
		for j in xrange(param['nr_cols']):
			maxw = -100
			map2X = 0; 
			map2Y = 0;
			for k in xrange(param['nr_rows']):
				for t in xrange(param['nr_cols']):
					if (hebb_weights[i][j][k][t] > maxw):
						maxw= hebb_weights[i][j][k][t];
						map2X = k; map2Y = t;
			string = "(" + str(map2X) + ", " + str(map2Y) + ")  "
			sys.stdout.write(string)
		print ''	
			
if __name__=="__main__":
	nr_pts = 1000
	path = get_path()

	# get the coordinates learned during random motor babbling 
	data = read_data(path, nr_pts)

	# train self-organizing maps
	som_hands = train_som(data['hands'])
	som_joints = train_som(data['joints'])
	
	print "Using %d data points for training"%(som_hands.data.shape[0])
	
	#plot(som_hands, som_joints)
	inact = som_hands.activation_response(som_hands.data)
	coord_inact = where(inact.flatten()==0)[0]
	print '%.0f%% of unactivated nodes'%(len(coord_inact)/product(som_hands.weights.shape[:2]) *100)
		
	#plot_inactivated_nodes(som_joints, coord_inact)
	plot_inactivated_nodes(som_hands, coord_inact)		
		
	# hebbian weights connecting maps
	hebb = hebbian_learning(som_hands, som_joints)

	#print_strongest_connections(hebb)	
	mse = 0
	_, w = som_joints.get_weights()
	unnorm = lambda x: x*som_joints.norm.ranges + som_joints.norm.mins
	
	for i in xrange(nr_pts):
		idx = random.randint(0, len(som_hands.data)-1) # l(sh.d) == l(sj.d)
		hands_view = som_hands.data[idx, :]
		
		# activate a neuron in the first map
		win_1 = som_hands.winner(hands_view)
		
		# find the neuron with the strongest connection in the second map
		win_2 = unravel_index(hebb[win_1[0], win_1[1], :, :].argmax(), \
			som_hands.weights.shape[:2])
			
		# get its weights
		joints = som_joints.weights[win_2[0], win_2[1], :]
		
		sim_joints, q = get_similar_data(w, som_joints.data[idx, :])
	
		if 0:
			print 'Data vector:', f(som_joints.data[idx, :])
			print 'Hebb chosen vector:', f(joints)
			#print 'Similar vectors:\n', 
			#for i, j in zip(sim_joints, q):
			#	print i, j
		
		dist = sum(som_joints.data[idx, :] - joints)
		mse += sum((som_joints.data[idx, :] - joints)**2)
		
	print 'MSE', mse/nr_pts	
	
	## Prepare data for .csv file to be processed by C++ program
	# Project weights into original data space
	wh = Normalizer(som_hands.get_weights()[1]).minmax()
	wh *= som_hands.norm.ranges
	wh += som_hands.norm.mins
	
	wj = Normalizer(som_joints.get_weights()[1]).minmax()
	wj *= som_joints.norm.ranges
	wj += som_joints.norm.mins
	
	# Save SOM 1
	savetxt('som1.csv', wh, delimiter=',')
	
	# Save SOM 2
	savetxt('som2.csv', wj, delimiter=',')
	
	# Save Hebbian weights 
	savetxt('hebb.csv', hebb.reshape(wh.shape[0], wj.shape[0]), delimiter=',')
	
	
	# Prediction flow (the most painless version): 
	# - get hand marker xyz coordinates
	# - normalize
	# - find the closest existing xyz point in KB  (!)
	# - fetch the corresponding joints from hebb weights
	# - send out motor command
	
