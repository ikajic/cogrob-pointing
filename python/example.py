import matplotlib.pyplot as plt
from plot_som import fetch_data, plot_weights, plot_data3d


base_path = '/home/ivana/knnl-0.1.4/knnl/build/make/data/'	

init_som = fetch_data(base_path + 'w_init_hands')
final_som = fetch_data(base_path + 'w_final_hands')

plot_weights(init_som, final_som)

data = fetch_data(base_path + 'babbling_KB_left_arm.dat', skip = 3)[::20, [2, 3, 4]]
plot_data3d(init_som, final_som, data, 'Motor babbling')


plt.show()
