import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from plot_som_dimensions import fetch_data, soms

data = fetch_data('../data/babbling_KB_left_arm.dat', skip = 3)[::20, [2, 3, 4]]
nr_nodes = 50

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(soms['init'][0][:nr_nodes,0], soms['init'][0][:nr_nodes,1], soms['init'][0][:nr_nodes,2], c='g', marker='o')
ax.plot(soms['final'][0][:nr_nodes,0], soms['final'][0][:nr_nodes,1], soms['final'][0][:nr_nodes,2], c='r', marker='o', alpha = 0.4)
ax.scatter(data[:, 0], data[:,1], data[:,2], c='b', alpha=0.2)

plt.show()


#scatter3(xhands(:,1), xhands(:,2),xhands(:,3)exit, 'b', 'filled')
