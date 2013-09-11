import pylab as pl
import numpy as np
import pdb

def fetch_data(path, skip = 0):
	with open(path) as file:	
		for _ in xrange(skip):
			print 'skipping some lines'
			next(file)
		rows = [filter(None, [x.strip() for x in line.split('\t')]) for line in file]
	data = np.array(rows).astype(float)
	return data

base_path = '../data/'	

soms = { 'init': [fetch_data(base_path + 'w_init_hands'), 
		 fetch_data(base_path + 'w_init_joints')],
	'final': [fetch_data(base_path + 'w_final_hands'),
		 fetch_data(base_path + 'w_final_joints')]}

if __name__=='__main__':	


	n, d = soms['init'][0].shape
	width = np.round(np.sqrt(n))

	# visualize weights from each dimension
	labs = ['x', 'y', 'z']
	titles = ['hands', 'joints']

	for t, title in enumerate(titles):
		pl.figure()
		pl.suptitle(title)
		for lab, i in zip(labs, xrange(d)):
			pl.subplot(2, 3, i+1)
			pl.title(lab)
			# plot weights before training
			pl.imshow(soms['init'][t][:,i].reshape(width, width))

			pl.subplot(2, 3, (i+1) + d)
			# same weights after training
			pl.imshow(soms['final'][t][:,i].reshape(width, width))
	
	pl.show()	
