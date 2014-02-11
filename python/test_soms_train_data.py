from __future__ import division

from scipy.stats import ttest_rel, ttest_ind
from minisom_nao import read_data
from dbg import *

import numpy as np
import cPickle as pickle
import random

import pdb

def generate_test_set(data, pts):
    """
    Random picking of *pts* data points from the *data* matrix.
    @param data: np.array
    @param pts: int
    """            
    test_set = np.asarray(random.sample(data, pts))
    
    return test_set 

def get_soms(path):
    """
    Fetch MiniSom instance stored in soms.pkl within a directory specified by the 
    path.
    @pram path: path to the directory 
    """
    with open('somconf/' + path + '/soms.pkl', 'rb') as inp:
        hands = pickle.load(inp)            
    return hands

def different_test_sets(spaths, nr_trials=1):
    """
    Use test sets with various number of samples to compute the prediction error.
    @param spaths: list of directories containing SOM configurations with soms.pkl files
    @param nr_trials: number of trials 
    """
    nr_pts = np.array([1, 10, 25])*1000                              
    
    soms = []
    for path in spaths:
        soms.append(get_soms(path))

    errors = np.zeros((len(soms), len(nr_pts), nr_trials))
    for i in xrange(nr_trials):
        for j, pts in enumerate(nr_pts):
            data = generate_test_set(test_data, pts)
            print pts, " points"
            for k in np.arange(len(spaths)):
                err = soms[k].quantization_error(data)
                print "SOM ", spaths[k], " error: ", err.mean()
                errors[k,j,i] = err.mean()
                
    return errors                
                
            
def prediction_errors(data, paths):    
    """
    For each SOM specified in paths compute the prediction error for each vector in *data*.
    Prediction error is the Euclidian distance between the vector and the weights of a winning 
    neuron in the SOM.
    @param data: np.array
    @param paths: list of directories containing SOM configurations with soms.pkl files
    """    
    
    #prettyprint stuff
    max_col = 20   
    print "".join(word.ljust(max_col) for word in ["Network conf","Mean error","STD error"])
    print '-'*max_col*3
    
    errors = np.zeros((len(paths), len(data)))
    
    for i, path in enumerate(paths):
        som = get_soms(path)
        errors[i, :] = som.quantization_error(data)   
        outputs = [path, str(errors[i, :].mean()), str(errors[i, :].std())]
        print "".join(word.ljust(max_col) for word in outputs)
        
    return errors     
    
def print_table(table):
    col_width = [max(len(str(x)) for x in col) for col in zip(*table)]
    for line in table:
        print "| " + " | ".join("{:{}}".format(str(x), col_width[i])
                                for i, x in enumerate(line)) + " |"
                   
def table_print_errors(errors, somconfs):
    """
    Run independent t-tests with every pair of SOMs specified by the list of
    directories in somconfs.
    @param errors: np.array
    @param somconfs: list of directories containing SOM configurations with soms.pkl files
    """
    print "-SOM index \t SOM conf "
    for i, label in enumerate(somconfs):
        print str(i).center(10),"\t",label

    table = np.zeros((len(somconfs)+1, len(somconfs)+1))
    
    for i in np.arange(1, table.shape[0]):
        table[i, 0] = table[0, i] = i-1
        for j in np.arange(i+1, table.shape[0]):
            err = "%.3f"%ttest_rel(errors[i-1, :], errors[j-1, :])[1]
            table[i,j] = err
    print 'T-test p-values'
    print_table(table.tolist())

if __name__=="__main__":
    test_path = "../data/r_29min_58k.dat"
    
    # 100 as argument means use 100% of data in *.dat folder
    test_data = read_data(test_path, 100, 'u')['hands']
    somconfs = ("5x5_70000_u100",
                "15x15_70000_u100")
                #"15x15_20000_f10", "15x15_20000_u100")
    
    #err1 = different_test_sets(somconfs, nr_trials=2)
    errors = prediction_errors(test_data, somconfs)    
    
    #table_print_errors(errors)

