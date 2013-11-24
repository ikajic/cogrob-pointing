from __future__ import division

from scipy.stats import ttest_rel
from minisom_nao import read_data
from dbg import *

import numpy as np
import cPickle as pickle
import random

def generate_test_set(data, pts):
    """
    Random picking of *pts* data points from the knowledge base.
    Using only hand coordinates
    """
            
    test_set = np.asarray(random.sample(data, pts))
    
    return test_set 

def get_soms(path):
    with open('somconf/' + path + '/soms.pkl', 'rb') as inp:
        hands = pickle.load(inp)            
    return hands

def different_test_sets(spaths, nr_trials=1):
    nr_pts = np.array([1, 10, 25])*1000                              
    soms = []
    for path in spaths:
        soms.append(get_soms(path))

    errors = np.zeros((len(soms), len(nr_pts), nr_trials))
    for i in range(nr_trials):
        for j, pts in enumerate(nr_pts):
            data = generate_test_set(test_data, pts)
            print pts, " points"
            for k in np.arange(len(spaths)):
                err = soms[k].quantization_error(data)
                print "SOM ", spaths[k], " error: ", err.mean()
                errors[k,j,i] = err.mean()
                
            
def different_training_sets(test_data, paths):    
    data = test_data
    
    errors = np.zeros((len(paths), len(data)))
    print "Network conf \t Mean error \t Error STD "
    print '-'*50
    for i, path in enumerate(paths):
        som = get_soms(path)
        errors[i, :] = som.quantization_error(data)        
        print "%s \t %.4f \t %.4f"%(path, errors[i, :].mean(), errors[i, :].std())
        
    return data, errors        
                

if __name__=="__main__":
    test_path = "../data/34min_64k.dat"
    test_data = read_data(test_path, 100, 'u')['hands']

    somconfs = ["5x5_20000_f1",
                "5x5_20000_f10",
                "5x5_20000_u10",
                "5x5_20000_u100",
                "10x10_20000_f1",
                "10x10_20000_f10",
                "10x10_20000_u10",
                "10x10_20000_u100",
                "15x15_20000_f1",
                "15x15_20000_f10",
                "15x15_20000_u10",
                "15x15_20000_u100"]

    #err1 = different_test_sets(somconfs, nr_trials=2)
    data, err = different_training_sets(test_data, somconfs)    

