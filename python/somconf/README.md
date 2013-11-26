### How to understand names of directories?

This folder contains trained SOM networks under different conditions.
These conditions are also reflected in the directory name of a format ```nxn_e_mode```.
Additional information on training can be found in `out.log` and `param.json` files within each folder.

* `nxn` is the size of a SOM. For example if a network as 5 columns and 5 rows, the overall number of neurons in that SOM is 25
* `e` is the number of epochs used to train a network
* `mode` string contains two parts (`lp`), a letter `l` which can be `f` or `u` and a number `p`. 
The letter defines how training data were chosen from a training set. 
`f` means only first n points were used, `u` means the data were uniformly chosen such that each n-th point was choosen. n is computed from the number `p` which denotes percentage.

For example, if the directory name is: `5x5_20000_f1` that means it contains configuration for an SOM with 25 neurons that was trained on 20000 epochs and only first 1% of data from a training set were used in training.
