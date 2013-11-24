import plot_som as ps
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt


def fetch_som(path):
    with open(path+'soms.pkl', 'r') as f:
        hands = pickle.load(f)
    return hands

def plot3(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot(data[:, 0], data[:,1], data[:,2])
    plt.show()    
    
    
def plot_and_save(som_hands, som_joints, savepath='', param=None, offset=50):
    """
    3D Plots of babblilng data and SOM sheets. If neurons have more than 3
    dimensions they are truncated to 3
    """
     
    wi_0, w_0 = som_hands.get_weights()    
    wi_1, w_1 = som_joints.get_weights()

    ps.plot_3d(final_som=w_0, data=som_hands.data[::offset, :], init_som=wi_0, nr_nodes=param['n_to_plot'], title='SOM Hands')
    if savepath: plt.savefig(savepath + 'som_hands.png')
    
    ps.plot_3d(final_som=w_1, data=som_joints.data[::offset, :], init_som=wi_1, nr_nodes=param['n_to_plot'], title='SOM Joints')
    if savepath: plt.savefig(savepath + 'som_joints.png')


def plot_inactivated_nodes(som, inact):
    """
    For a given SOM plots neurons and marks the ones which are never
    activated in green
    """
    _, w = som.get_weights()    
    act = np.setdiff1d(np.arange(w.shape[0]), inact)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    data = som.data
    
    # Just a fraction of data used for plotting
    data = data[:,:3][::20]
    d = ax.plot(data[:, 0], data[:,1], data[:,2], c='b', marker='*', linestyle='None', alpha=0.4, label='data')
    
    # Plot activated nodes in green
    act_nod = w[act, :]
    a = ax.plot(act_nod[:, 0], act_nod[:, 1], act_nod[:, 2], c='g', marker='o', alpha = 0.6, label='neurons', markersize=4)
    
    # Plot inactivated nodes in red
    in_w = w[inact, :]
    i = ax.plot(in_w[:, 0], in_w[:, 1], in_w[:, 2], c='r', marker='o', alpha = 0.6, label='inact. neurons', markersize=6, linestyle='None')
    plt.legend(numpoints=1)
    
def print_strongest_connections(hebb_weights):
    # print the strongest connections
    for i in xrange(param['nr_rows']):
        for j in xrange(param['nr_cols']):
            maxw = -100
            map2X = 0; 
            map2Y = 0;
            for k in xrange(param['nr_rows']):
                for t in xrange(param['nr_cols']):
                    if (hebb_weights[i][j][k][t] > maxw):
                        maxw= hebb_weights[i][j][k][t];
                        map2X = k; map2Y = t;
            string = "(" + str(map2X) + ", " + str(map2Y) + ")  "
            sys.stdout.write(string)
        print ''    
        

def dbg_print_inact(som_hands, som_joints, defstream):
    inact = som_hands.activation_response(som_hands.data)
    coord_inact = np.where(inact.flatten()==0)[0]
    defstream.write("Hands: %.0f%% of unactivated nodes\n"%(len(coord_inact)/np.product(som_hands.weights.shape[:2]) *100))
    
    inact = som_joints.activation_response(som_joints.data)
    coord_inact = np.where(inact.flatten()==0)[0]
    defstream.write("Joints: %.0f%% of unactivated nodes\n"%(len(coord_inact)/np.product(som_joints.weights.shape[:2]) *100))
    
    
def validate(som_hands, som_joints, hebb, data, frac):

    assert len(data['hands'])==len(data['joints'])
    
    _, joints_weights = som_joints.get_weights()
    unnorm = lambda x: x*som_joints.norm.ranges + som_joints.norm.mins    
      
    data['hands'] = (data['hands']-som_hands.norm.mins)/som_hands.norm.ranges    
    data['joints'] = (data['joints']-som_joints.norm.mins)/som_joints.norm.ranges    
    
    mse = np.zeros((data['joints'].shape))
    
    nr_pts = np.asarray(
        random.sample(np.arange(len(data['hands'])), 
        int(frac*len(data['joints'])))
        )

    for idx in nr_pts:
        # activate a neuron in the first map
        test_hands = data['hands'][idx, :] 
        test_joints = data['joints'][idx, :]
        win_1 = som_hands.winner(test_hands)
        
        # activate a neuron in the second map based on the strongest connection
        win_2 = unravel_index(hebb[win_1[0], win_1[1], :, :].argmax(), \
            som_hands.weights.shape[:2])            
        node_joints = som_joints.weights[win_2[0], win_2[1], :]
        
        node_joints, q, _ = get_similar_vector(joints_weights, test_joints) 
        mse += (test_joints - node_joints)**2
        
    return (mse/len(nr_pts)).mean(axis=0)    
