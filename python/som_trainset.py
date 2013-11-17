from __future__ import division

from scipy.stats import ttest_ind
from minisom_nao import read_data

import numpy as np
import cPickle as pickle
import random

def generate_test_set(som, nrpoints):
    """
    Random picking of nrpoints data points from the knowledge base.
    Using only hand coordinates
    """
    #dpath = '../data/babbling_KB_left_arm.dat'
    #data = read_data(dpath)
    
    test_set = np.asarray(random.sample(som.data, nrpoints))
    
    return test_set 
    

if __name__=="__main__":
    spaths = ['11_17-19_19_10/', '11_17-19_55_52/', '11_17-20_01_49/']
    nr_pts = np.array([1, 10, 25])*1000
    
    # read in all three soms
    soms = []
    for path in spaths:
        with open('somconf/' + path + 'soms.pkl', 'rb') as inp:
            # read in only hands
            soms.append(pickle.load(inp)) 

    # generate test data from the first som
    nr_rep =  10
    errors = np.zeros((len(soms), len(nr_pts), nr_rep))
    for i in range(nr_rep):
        for j, pts in enumerate(nr_pts):
            data = generate_test_set(soms[0], pts)
            print pts, " points"
            for k, som in enumerate(soms):
                predicted = som.quantization(data)
                err = ((data-predicted)**2).sum()/pts
                print "SOM: ", som.weights.shape[:2], " error: ", err 
                errors[k,j,i] = err
