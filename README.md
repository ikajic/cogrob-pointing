### Nao robot learns how to point to an object


We used a humanoid robot Nao and a biologically plausible framework to understand the 
development of pointing behavior from manual babbling.
The robot explored the environment in the motor babbling phase where he randomly moved his hands and joints, 
followed by his gaze. Configurations of his joints were saved in the knowledge base.
Knowledge base was used to train two self-organizing maps (SOMs), one for 3D hand coordinates and one for 4D 
joint coordinates (shoulder pitch, shoulder roll, elbow yaw and elbow roll). 
The weigths between these two maps were learned using the Hebbian learning paradigm. 

The structure of the project is following:
* **c++** contains the code used to read the knowledge base, network weights and set  
joint configurations during pointing. This is the code ported to Nao.

* **py** contains the code used to read the knowledge base and train different configurations of self organizing maps. It uses networks from a modified version of [minisom](https://github.com/JustGlowing/minisom) implementation.

* **experiments** contains data obtained during the experiments with Nao.
How to read the `.csv` header:
```
pt, ht_x, ht_y, ht_z, hp_x, hp_y, hp_z, dist, jt_x, jt_y, jt_z, jt_q, jp_x, jp_y, jp_z, jp_q ,dist
```
  * pt: is pointing type (0=beginner, 1=intermediate, 2=advanced)
  * ht_x, ht_y, ht_z: Nao's hand coordinates in the experiment (true)
  * hp_x, hp_y, hp_z: coordinates of the node with highest activation in the first SOM (predicted)
  * dist: euclidian distance between true and predicted hand coordinates
  * jt_x, jt_y, jt_z: Nao's joint coordinates from the knowledge base (true)
  * jp_x, jp_y, jp_z: coordinates of the node with the highest activation value in the second SOM (predicted)
  * dist: euclidian distance between true and predicted joint coordinates
