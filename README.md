# ClusterAttack
A clustering system that explores the success of attackers on defeating networks with frequent IP address changes to escape DoS attacks.
To run an experiment, you should decide on a few parameters: the name of a dataset to experiment with, the value of N (number of addresses to predict in each iteration), a value in [0,1] to decide the number of clusters (the value is multiplied by the size of the dataset), a limit on the number of IP addresses to include in the dataset.
For example, one can run an experiment this way: python3 driver.py US-EAST-1 5 0.05 -1 
Note that -1 as the last parameter indicates to include all IP addresses in the dataset. Currently, the code runs slowly if N<5. 
The script will do a clustering and three attack experiments. The results will output on files. 
