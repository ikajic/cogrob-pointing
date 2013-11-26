from __future__ import division

from numpy import genfromtxt, zeros, product, setdiff1d, arange, where,  unravel_index, tanh, savetxt
from parameters import param
from random import choice
from time import strftime

import cPickle as pickle
import json
import pdb
import os
import pdb
import random
import shutil
import sys

from minisom import MiniSom, Normalizer
from similar_vec import get_similar_vector
from dbg import *


def read_data(path, p_pts, mode='u'):
    """
    Return babbling coordinates for hands and joints
    """
    
    cols_hands = param['hands']
    cols_joints = param['joints']
        
    with open(path) as f:
        hands = genfromtxt(f, dtype=np.float, skip_header=2, usecols=cols_hands)
    
    with open(path) as f:    
        joints = genfromtxt(f, dtype=np.float, skip_header=2, usecols=cols_joints)
    
    if mode=='f':        
        data = {
            'hands': hands[:len(hands)*p_pts/100, :],
            'joints': joints[:len(joints)*p_pts/100, :]
            }
    elif mode=='u':
        data = {
        'hands': hands[::(100/p_pts), :],
        'joints': joints[::(100/p_pts), :]
        }            

    return data 

def train_som(data, offset=None):
    """
    offset: offset between points used for training
    """
    
    if offset:
        data = data[::offset, :]
    
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
    
    # minisom uses distances as activations, here we use sigmoid over distances
    # to get activations for hebbian learning
    f = lambda x: 1/(1+tanh(x))
    for dp1, dp2 in zip(som1.data, som2.data):
        act1 = som1.activate(dp1)
        act2 = som2.activate(dp2)
                
        idx1 = som1.winner(dp1)
        idx2 = som2.winner(dp2)
        
        hebb[idx1[0], idx1[1], idx2[0], idx2[1]] += param['eta'] * f(act1[idx1]) * f(act2[idx2])
        
    return hebb

            
def makeDirs(dirname):
    basePath = '../python/somconf/' + dirname + '/'
    overwrite = True
    
    if os.path.exists(basePath):
        inp = raw_input('Directory %s already exists. Overwrite? [y/n]: '%dirname)
        if inp!='y': 
            overwrite = False
            basePath = None        
            
    if overwrite:    
        try:
            os.makedirs(basePath)
        except OSError:
            shutil.rmtree(basePath)        
            os.makedirs(basePath)

    return basePath      
            
            
def parse_cmd_inp():
    nrarg = len(sys.argv)

    if nrarg<5:
        raise Exception('Please provide all arguments: save [0,1], net_size (5,10,15), nr_epochs, mode [f(irst), u(niform) + p(roportion of test data used)]')

    save_nets = int(sys.argv[1])
    n_size = int(sys.argv[2])
    epochs = int(sys.argv[3])
    mode = str(sys.argv[4][0])
    p_train = int(sys.argv[4][1:])
    
    return save_nets, n_size, epochs, mode, p_train

            
if __name__=="__main__":
    savepath = ''
    defstream = sys.stdout
    save_nets, n_size, epochs, mode, p_train = parse_cmd_inp()
    param['nr_epochs'] = epochs
    param['nr_rows'] = n_size
    param['nr_cols'] = n_size
    
    if save_nets: 
        dirname = "%dx%d_%d_%s%d"%(n_size, n_size, epochs, mode, p_train)   
        savepath = makeDirs(dirname)
        if (savepath):       
            defstream = open(savepath + 'out.log', 'w')
        else:
            print "Ok, I ain't savin' anythin!"
            save_nets = 0
        
    train_path = '../data/r_37min_74k.dat'
    train_data = read_data(train_path, p_train, mode)

    som_hands = train_som(train_data['hands'])
    som_joints = train_som(train_data['joints'])
    
    defstream.write("Using %d (%s) data points for training. \n" % (som_hands.data.shape[0], mode))
    plot_and_save(som_hands, som_joints, savepath, param, offset=1)
    print "Done plotting!"
    
    dbg_print_inact(som_hands, som_joints, defstream)      
        
    # hebbian weights connecting maps
    hebb = hebbian_learning(som_hands, som_joints)
    defstream.write("Learned Hebbian weights.\n")
    
    if save_nets:
        defstream.close()        
        savestr = zip(['som1.csv','som2.csv', 'hebb.csv'],
                       [som_hands.get_weights()[1], 
                        som_joints.get_weights()[1], 
                        hebb.reshape(n_size*n_size, n_size*n_size)])

        # Save weights
        for fname, dat in savestr:
            savetxt(savepath+fname, dat, delimiter=',')
    
        # Save simulation parameters
        with open(savepath + 'param.json', 'w') as f:
            json.dump(param, f)
        
        with open(savepath + 'soms.pkl', 'wb') as output:
            som_hands.neighborhood = None
            som_joints.neighborhood = None
            
            pickler = pickle.Pickler(output, -1)
            pickle.dump(som_hands, output)
            pickle.dump(som_joints, output)
