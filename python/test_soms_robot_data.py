import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel

def plot_ed_over_time(coords, labs):
    err_idx = [7]
    err_lab = ['hands']
    
    d_avg = lambda x: np.ones(x.shape[0])*np.mean(x[:, err_idx])  	   
    
    avgs = []
    for v in coords.values():
        avgs.append(d_avg(v))

    plt.figure()
    plt.title('Euclidian distances for predicted and true hand coordinates')
    cols = ['b', 'r', 'g', 'm']
    
    for i, (k,v) in enumerate(coords.iteritems()):
        plt.plot(v[skip_begin:, err_idx], color=cols[i], label=k)
        plt.plot(avgs[i][skip_begin:], color=cols[i], alpha=0.3)    	    
    
#    for i, k in enumerate(coords.keys()):


    plt.xlabel('Time [0.5 s]')
    plt.legend()

if __name__=="__main__":

    exp_path = "../experiments/nao/bigger_map2/out_"

    # Load coordinates obtained during the pointing experiment
    labs = ['b100', 'b10', 's100', 's10']
    coords = {c: np.genfromtxt(exp_path + c + '.csv', dtype=np.float, skiprows=1, delimiter=',') for c in labs}
  
    print 'Avg errors and stds (before normalization)'
    for k in labs:
        print k, np.mean(coords[k][:, 7]), np.std(coords[k][:, 7]) 
    print '\n'
             
    # Hands euclidian distances: column 7
    # trim array if required (usually in the beginning or at the end when some
    # wild values appear)
    skip_begin = 10
    for k in labs:
        coords[k] = coords[k][skip_begin:,]
        
    #plot_ed_over_time(coords, labs)    

    # normalize    
    max_err =  np.max([np.max(coord[:,7]) for coord in coords.values()])
    for k in labs:
        coords[k][:,7] = coords[k][:,7]/max_err
    
    plot_ed_over_time(coords, labs)

    print 'Avg errors and stds (after normalization)' 
    for k in labs:
        print k, np.mean(coords[k][:, 7]), np.std(coords[k][:, 7]) 
    print '\n'
    
    for lab1 in labs:
        for lab2 in labs:
            if lab1==lab2: continue
            print lab1, ' ', lab2, ': ', ttest_rel(coords[lab1][:, 7], coords[lab2][:, 7])
    
    plt.savefig(exp_path + "error_hands.png")
    plt.show()
