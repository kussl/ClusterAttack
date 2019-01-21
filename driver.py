import sys,pickle
import statistics as stats 
from data import DataSource
from simulator import Simulator
from math import floor
import matplotlib.pyplot as plt
from matplotlib import rcParams,rc

#rcParams['font.family'] = 'STIXGeneral'
#rcParams['font.sans-serif'] = ['STIXGeneral']
rcParams['font.size']= 18

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

#Clustering and predicting using a single account.
def cluster3(region,N,kp,limit):
	CL = Simulator()
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

	failed, counts1 = CL.simulate_attack(X_train,X_test,T,clusters,region,U)

	failed, counts2 = CL.simulate_random_attack(X_train,X_test,region,U,maxtrials=floor(stats.mean(counts1)))

	F = CL.load_unique_data_with_frequencies(region,N,limit=len(X_train)*N)

	failed, counts3 = CL.simulate_frequency_attack(X_train,X_test,region,F,maxtrials=floor(stats.mean(counts1)))

	#plt.figure(figsize=(5,5))
	plt.tight_layout() 
	plt.boxplot([counts1,counts3,counts2], meanline=True, showcaps=True, whis='range', widths=0.3)

	#plt.legend(['Cluster','Random','Frequency'],loc=1, prop={'size': 8})
	#plt.ylabel('Number of guesses to predict N Prefixes',wrap=True)
	#plt.xlabel('Attack strategy')
	plt.xticks([1,2,3],['Cluster','Frequency','Random'])
	plt.savefig('results_'+region+'.pdf',format='pdf')

	f = open('allcounts_'+region,'w')
	print(counts1,file=f)
	print(counts3,file=f)
	print(counts2,file=f)
	f.close()

#Clustering and predicting using two different user accounts
def cluster4(region,N,kp,limit):
	CL = Simulator()
	DS = DataSource() 

	print('Attacking from two different accounts...')
	print('limit:', limit)

	dataset_A = CL.load_data(region+'-A',N,limit)
	n = len(dataset_A)
	dataset_B = CL.load_data(region+'-B',N,limit) 
	m = len(dataset_B)

	if n < m: 
		dataset_B= dataset_B[:n]
	else: 
		dataset_A= dataset_A[:m]

	n = len(dataset_A)

	print('Dataset size: ', n)

	#A cutoff to separate train and test sequences. 
	train_cutoff = int(n*0.70)

	X_train = dataset_A
	X_test  = dataset_B

	U = CL.unique_combinations(X_train)

	print('Unique values: ', len(U))

	#Choose the size of clusters.
	K= int(kp*len(X_train))

	print('Size of train set', len(X_train), 'and the test set', len(X_test), 'K:',K)
	clusters,T = CL.hausdorffcluster(X_train,K,N)
	store_clusters(X_train,clusters,region)
	store_matrix(T,region)

	failed, counts1 = CL.simulate_attack(X_train,X_test,T,clusters,region,U)

	failed, counts2 = CL.simulate_random_attack(X_train,X_test,region,U,maxtrials=floor(stats.mean(counts1)))

	F = CL.load_unique_data_with_frequencies(region,N,limit=len(X_train*N))

	failed, counts3 = CL.simulate_frequency_attack(X_train,X_test,region,F,maxtrials=floor(stats.mean(counts1)))

	#plt.figure(figsize=(15,15))
	#plt.tight_layout() 
	plt.boxplot([counts1,counts3,counts2], meanline=True, showcaps=True,  whis='range')

	#plt.legend(['Cluster','Random','Frequency'],loc=1, prop={'size': 8})
	#plt.ylabel('Number of guesses to predict N Prefixes',wrap=True)
	#plt.xlabel('Attack strategy')
	plt.xticks([1,2,3],['Cluster','Frequency','Random'])
	plt.savefig('results_'+region+'.pdf',format='pdf')

	f = open('count_c_'+region,'wb')
	pickle.dump(counts1,f)
	f.close()

	f = open('count_r_'+region,'wb')
	pickle.dump(counts3,f)
	f.close()

	f = open('count_f_'+region,'wb')
	pickle.dump(counts2,f)
	f.close()

def driver(datasource='US-EAST-1',N=5,kp=0.05,datalimit=-1,attack_type='-single'):
	print('Dataset',datasource)
	print('Data limit:', datalimit, 'KP', kp, 'N', N, 'Attack Type', attack_type)
	if attack_type == '-single':
		cluster3(datasource,N,kp,datalimit)
	elif attack_type == '-dual':
		cluster4(datasource,N,kp,datalimit) 

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

if len(sys.argv) > 5: 
	attack_type = sys.argv[5]
else: 
	attack_type = '-single'


driver(datasource,N=N,kp=kp,datalimit=datalimit,attack_type=attack_type)