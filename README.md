### Using humanoid robot Nao to investigate the cognitive development of pointing behavior


We used a humanoid robot Nao and a biologically plausible framework for investigation 
of pointing behavior. 
First, the robot explored the environment through random motor babbling. 
The sensory values such as hand and joint coordinates were stored in a knowledge base.
These values were used to train self-organizing maps (SOMs) and we learned weights between them using 
Hebbian learning paradigm. 

The structure of the code is following:
* **c++** contains the code used to read the knowledge base, network weights and set  
joint configurations during pointing. This is the code ported to Nao.

* **py** contains the code used to read the knowledge base and train different configurations of self organizing maps. It uses networks from a modified version of [minisom](https://github.com/JustGlowing/minisom) implementation.

* **experiments** contains data obtained during the experiments with Nao.
