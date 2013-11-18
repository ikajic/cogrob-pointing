import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

# TODO: use numpy arrays to read in and handle columns from csv files

# Find the closest joint positions and their distances to predicted positions
def update_table(missing):
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

if __name__=="__main__":

    basepath = "../experiments/new_soms_3sec_adv/"
    kbpath = "/home/ivana/babbling_KB_left_arm.dat"

    # Load coordinates obtained during the pointing experiment
    b = np.genfromtxt(basepath + 'out_b.csv', dtype=np.float, skiprows=1, delimiter=',')
    i = np.genfromtxt(basepath + 'out_i.csv', dtype=np.float, skiprows=1, delimiter=',')
    a = np.genfromtxt(basepath + 'out_a.csv', dtype=np.float, skiprows=1, delimiter=',')

    # Load the knowledge base from the babbling phase
    with open(kbpath) as dat:
	    kb = np.genfromtxt(dat, dtype=np.float, skip_header=2, usecols=(2,3,4,7,8,9,10))

    if 0:
        b = update_table(b)
        a = update_table(a)
        i = update_table(i)
	
	
    # Hands euclidian distances: column 7
    # Joints euclidian distances: column 16
    skip_begin = 0
    a = a[skip_begin:, :]
    b = b[skip_begin:, :]
    i = i[skip_begin:, :]
    for idx, lab in zip([7, 16], ['hands', 'joints']):
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

    print 'BEG-INT', ttest_ind(b[:, 7], i[:, 7])
    print 'INT-ADV', ttest_ind(i[:, 7], a[:, 7])
    print 'ADV-BEG', ttest_ind(a[:, 7], b[:, 7])

    plt.show()
