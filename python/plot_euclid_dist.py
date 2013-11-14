import numpy as np
import matplotlib.pyplot as plt

b = np.genfromtxt('out_b.csv', dtype=np.float, skiprows=1, delimiter=',')
i = np.genfromtxt('out_i.csv', dtype=np.float, skiprows=1, delimiter=',')
a = np.genfromtxt('out_a.csv', dtype=np.float, skiprows=1, delimiter=',')

# Load the knowledge base from the babbling phase
path = '/home/ivana/babbling_KB_left_arm.dat'
with open(path) as dat:
	kb = np.genfromtxt(dat, dtype=np.float, skiprows=2, usecols=(2,3,4,7,8,9,10))

# Find the closest joint positions and their distances to predicted positions
def create_all(missing):
	f_all = np.zeros((len(missing), 17), dtype=np.float)
	for i, row in enumerate(missing):
		hands = row[1:4]	
		joints = row[9:13]
		distances = map(lambda x: np.linalg.norm(x-hands), kb[:, :3])	
		idx = np.argmin(distances)
		f_all[i, :] = np.r_[row[:8], 
							kb[idx, 3:], 
							joints, 
							np.linalg.norm(kb[idx, 3:]-joints)]
							
	return f_all

b = create_all(b)
a = create_all(a)
i = create_all(i)
	
for idx, lab in zip([7, 16], ['hands', 'joints']):

	# hands column 7
	# joints column 16

	d_avg = lambda x: np.ones(x.shape[0])*np.mean(x[:, idx]) # euclid dist is at pos 8
	avg_b = d_avg(b)
	avg_i = d_avg(i)
	avg_a = d_avg(a) 

	plt.figure()
	plt.title('Euclidian distances for predicted and true ' + lab + ' coordinates')
	plt.plot(a[:, idx], color='b', label='adv')
	plt.plot(avg_a, color='b', alpha=0.3)

	plt.plot(b[:, idx], color='r', label='begin')
	plt.plot(avg_b, color='r', alpha=0.3)

	plt.plot(i[:, idx], color='g', label='int')
	plt.plot(avg_i, color='g', alpha=0.3)

	plt.xlabel('Time [0.5 s]')
	plt.legend()
	plt.show()

plt.show()
