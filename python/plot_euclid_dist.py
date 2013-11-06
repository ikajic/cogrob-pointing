import numpy as np
import matplotlib.pyplot as plt

f = np.genfromtxt('out.csv', dtype=np.float, skiprows=1, delimiter=',')


# Load the knowledge base from the babbling phase
path = '/home/ivana/babbling_KB_left_arm.dat'
with open(path) as dat:
	kb = genfromtxt(dat, dtype=np.float, skiprows=2, usecols=(2,3,4,7,8,9,10))

# Find the closest joint positions and their distances to predicted positions
def create_all():
	f_all = np.zeros((len(f), 17), dtype=np.float)
	for i, row in enumerate(f):
		hands = row[1:4]	
		joints = row[9:13]
		distances = map(lambda x: np.linalg.norm(x-hands), kb[:, :3])	
		idx = argmin(distances)
		f_all[i, :] = np.r_[row[:8], 
							kb[idx, 3:], 
							joints, 
							np.linalg.norm(true_joint-joints)]

b = f[np.where(f[:,0]==0), :][0]
i = f[np.where(f[:,0]==1), :][0]
a = f[np.where(f[:,0]==2), :][0]

for idx, lab in zip([7, 16], ['hands', 'joints']):

	# hands column 7
	# joints column 16

	d_avg = lambda x: np.ones(x.shape[0])*np.mean(x[:, idx]) # euclid dist is at pos 8
	avg_b = d_avg(b)
	avg_i = d_avg(i)
	avg_a = d_avg(a) 

	#plt.figure()

	plt.figure()
	plt.title('Euclidian distances for predicted and true ' + lab + ' coordinates')
	plt.plot(a[:, idx], color='b', label='adv')
	plt.plot(avg_a, color='b', alpha=0.3)

	plt.plot(b[:, idx], color='r', label='begin')
	plt.plot(avg_b, color='r', alpha=0.3)

	plt.plot(i[:, idx], color='g', label='int')
	plt.plot(avg_i, color='g', alpha=0.3)
	plt.ylim([0,300])
	plt.xlim([0,250])

	plt.xlabel('Time [0.5 s]')
	plt.legend()

plt.show()
