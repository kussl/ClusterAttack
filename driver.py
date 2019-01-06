import sys
import statistics as stats 
from data import DataSource
from centroid import CentroidCluster
from math import floor

def store_clusters(dataset,clusters,region):
	f = open('clusters_'+region+'.txt','w')
	for cluster in clusters: 
		print(id(cluster),file=f)
		for i in cluster:
			print(dataset[i],i,file=f)

	f.close()

def store_matrix(T,region): 
	f = open('T_'+region+'.txt','w')
	for i in range(len(T)): 
		print(i,T[i],file=f)
	f.close()

def cluster3(region,N,kp,limit):
	CL = CentroidCluster()
	DS = DataSource() 
	dataset = CL.load_data(region,N,limit) 
	n = len(dataset)

	print('Dataset size: ', n)

	#A cutoff to separate train and test sequences. 
	train_cutoff = int(n*0.70)

	X_train = dataset[:train_cutoff]
	X_test  = dataset[train_cutoff:]

	U = CL.unique_combinations(X_train)

	print('Unique values: ', len(U))

	#Choose the size of clusters.
	K= int(kp*len(X_train))

	print('Size of train set', len(X_train), 'and the test set', len(X_test), 'K:',K)
	clusters,T = CL.hausdorffcluster(X_train,K,N)
	store_clusters(X_train,clusters,region)
	store_matrix(T,region)

	failed, counts = CL.simulate_attack(X_train,X_test,T,clusters,region,U)

	failed, counts = CL.simulate_random_attack(X_train,X_test,region,U,maxtrials=floor(stats.mean(counts)))

def driver(datasource='US-EAST-1',N=5,kp=0.05,datalimit=-1):
	print('Dataset',datasource)
	print('Data limit:', datalimit, 'KP', kp, 'N', N)
	cluster3(datasource,N,kp,datalimit)

datasource = sys.argv[1]
if len(sys.argv) > 2: 
	N = int(sys.argv[2])
else: 
	N = 5 
if len(sys.argv) > 3: 
	kp = float(sys.argv[3])
else: 
	kp = 0.05 
if len(sys.argv) > 4: 
	datalimit = int(sys.argv[4])
else: 
	datalimit = -1 

driver(datasource,N,kp,datalimit)