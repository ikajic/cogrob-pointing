import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel

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

    basepath = "../experiments/nao/"

    # Load coordinates obtained during the pointing experiment
    csvs = ['out_b100', 'out_b10', 'out_s100', 'out_s10']
    coords = [np.genfromtxt(basepath + c + '.csv', dtype=np.float, skiprows=1, delimiter=',') for c in csvs]
  
    # Hands euclidian distances: column 7
    # Joints euclidian distances: column 16
    skip_begin = 0
    for i in range(len(coords)):
        coords[i] = coords[i][skip_begin:, :]        
    
    err_idx = [7]
    err_lab = ['hands']
    for idx, lab in zip(err_idx, err_lab):
	    d_avg = lambda x: np.ones(x.shape[0])*np.mean(x[:, idx])  	   
	    
	    avgs=[]
	    for i in range(len(coords)):
	        avgs.append(d_avg(coords[i]))

	    plt.figure()
	    plt.title('Euclidian distances for predicted and true ' + lab + ' coordinates')
	    cols = ['b', 'r', 'g', 'm']
	    
	    for i in range(len(coords)):
	        plt.plot(coords[i][skip_begin:, idx], color=cols[i], label=csvs[i])
	    
	    for i in range(len(coords)):
	        plt.plot(avgs[i], color=cols[i], alpha=0.3)    	    

	    plt.xlabel('Time [0.5 s]')
	    plt.legend()

    print 'Avg errors and stds'
    for i in range(len(coords)):
        print csvs[i], np.mean(coords[i][:, 7]), np.std(coords[i][:, 7]) 
        
    '''    
    print 'BEG-INT', ttest_rel(b[:, 7], i[:, 7])
    print 'INT-ADV', ttest_rel(i[:, 7], a[:, 7])
    print 'ADV-BEG', ttest_rel(a[:, 7], b[:, 7])
    '''
    plt.savefig(basepath+"error_hands.png")
    plt.show()
