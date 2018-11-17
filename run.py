import numpy as np
import csv,sys,random,math
import matplotlib.pyplot as plt
import statistics as stats 
import sklearn.preprocessing as prp 
from scipy.linalg import norm
from crossval import CrossValidator  
from sklearn.model_selection import train_test_split
from datetime import datetime 
from data import DataSource
from kmeans import KMeans
from ds import Dataset, Assignment
import statistics
from numpy import corrcoef
from scipy.linalg import norm
from plots import scatter_plot


def IPdata(source='IPdata.csv',upto=2):
   try:
      f = open(source,'r')
      reader = csv.reader(f,delimiter=',')
   except Exception as ex: 
      DS = DataSource()
      reader = DS.getrawdata(collection=source)
   
   addresses = []
   for row in reader:
      v=tuple(map(int,row[:upto]))
      addresses.append(v)
   return addresses 

def processIPData(addresses,N=5):
   #Split the IP addresses into arrays of size N. 
   X=np.array_split(addresses,len(addresses)//N)

   #Create a dataset object and 
   #create a new assignment for each array. 
   dataset = Dataset(assignments=[],N=N)
   if len(dataset) > 0:
   	  raise Exception('Dataset is not empty! There must be a bug somewhere in how the Dataset object works.')
      #return dataset
   k = 0 #order of assignment
   for x in X: 
      dataset.assign(Assignment(addresses=x,k=k))
      k+=1 
   return dataset

def retrieve_data(sourcefile,N):
	S=IPdata(source=sourcefile,upto=4)
	dataset = processIPData(S,N=N)
	print('Data points retrieved:', len(dataset),'x',N)
	return dataset

def dump_results(clusters,F,score,ext='',AS=0):
	f = open('results'+ext, 'w')
	print('Results: ',file=f)
	print('>ceil(N/2)', score, file=f)
	print('1..N', F,file=f)
	if AS > 0: print('Assignments used', AS,file=f)
	print('Points in clusters: ', sum([len(cluster.assignments) for cluster in clusters]), file=f)
	f.close()
	f = open('cluster_acc'+ext,'w')
	for cluster in clusters:
		print(cluster.cid, cluster.accuracy,file=f)
	f.close()


def driver_cluster_leading_bits(dataset,N=10,kp=0.1,test=False,reportclusters=False,min_acc=0.5,mode=0):
	#Stores the quality reports for the clusters.
	scores = [] 

	n = len(dataset)

	#A cutoff to separate train and test sequences. 
	train_cutoff = int(n*0.75)
	
	X_train = Dataset(assignments=dataset[:train_cutoff],N=N)
	X_test  = Dataset(assignments=dataset[train_cutoff:],N=N)
	print('Size of train set', len(X_train), 'and the test set', len(X_test))

	#Choose the size of clusters.
	K= int(kp*len(X_train))
	print('Requesting K:',K, 'size of training set is',train_cutoff)

	CI = KMeans()
	clusters,score,T,F = CI.cluster(X_train,X_test,K,N,min_acc=min_acc,mode=mode)
	dump_results(clusters,F,score)
	print('Final no of clusters:',len(clusters))
	scatter_plot(clusters,'dataset',N,score,bytes=[1,2])
	return clusters,T,X_test



#This function is ONLY for testing. Don't take it serious. 
'''
1. Take clusters that were computed for the leading 16 bits.
2. Collect all the assignments in a NEW dataset. 
3. Run the clustering over the entire data set, only taking the second and the third octets.
'''
def driver_cluster_third_eight_bits(data,clusters,N,kp,min_acc,f,s,ext):
	print('Now clustering the next level..')
	dataset = Dataset(assignments=[],N=N)

	for P in data:
		x = np.array([np.array([x[f],x[s]]) for x in P.addresses])
		y = np.array([np.array([x[0],x[1],x[2],x[3]]) for x in P.addresses])
		dataset.assign(Assignment(addresses=x,k=P.k,full_addresses=y))


	n = len(dataset)
	print('Dataset is ready:', n)

	#A cutoff to separate train and test sequences. 
	train_cutoff = int(n*0.75)
	
	X_train = Dataset(assignments=dataset[:train_cutoff],N=N)
	X_test  = Dataset(assignments=dataset[train_cutoff:],N=N)
	print('Size of train set', len(X_train), 'and the test set', len(X_test))

	#Choose the size of clusters.
	K= int(kp*len(X_train))
	print('Requesting K:',K, 'size of training set is',train_cutoff)

	CI = KMeans()
	clusters,score,T,F = CI.cluster(X_train,X_test,K,N,min_acc=min_acc,mode=0)

	dump_results(clusters,F,score,ext)
	print('Final no of clusters:',len(clusters))
	scatter_plot(clusters,'dataset',N,score,bytes=[f+1,s+1])
	return clusters,T,X_test

'''
This driver, develops three cluster sets and performs ONE complete 
cross validation for ALL predictions of the full IP address. 
'''
def driver_cluster_full_address(dataset,N,kp,min_acc,mode):
	clusters_set = [] 
	T_set = []
	X_tests = []
	P_set = [] 
	clusters,T,X_test = driver_cluster_leading_bits(dataset,N,kp,min_acc=min_acc,mode=mode)
	print('Results 1 recorded.')
	clusters_set.append(clusters)
	T_set.append(T)
	X_tests.append(X_test)
	
	clusters,T,X_test = driver_cluster_third_eight_bits(dataset,clusters,N,kp,min_acc*.9,1,2,'-2')
	clusters_set.append(clusters)
	T_set.append(T)
	print('Results 2 recorded.')

	print('Hierarchical cross validation:')
	PR = CrossValidator()
	clusters,F,AS = PR.hierarchical_cross_validate(X_tests,clusters_set,T_set,N)
	score = round(sum(F[math.ceil(N/2):]),3)
	dump_results(clusters,F,score,'-all',AS=AS)
	

def correlation(dataset,i,j):
	X = []
	Y = [] 
	for P in dataset: 
		X.append((P.addresses[0][i]))
	for P in dataset: 
		Y.append(P.addresses[0][j])

	cc = corrcoef(X,Y)
	return cc 



def driver():
	if len(sys.argv) > 1: 
		sourcefile=sys.argv[1]
	else:
		sourcefile = 'data_aws_ireland.csv' #Default set. 
	if len(sys.argv) > 2: 
		N=int(sys.argv[2])
	else:
		N=5 
	if len(sys.argv) > 3: 
		kp=float(sys.argv[3])
	else:
		kp=0.05
	if len(sys.argv) > 4: 
		min_acc = float(sys.argv[4])
	else:
		min_acc = 0.2
	if len(sys.argv) > 5: 
		mode = int(sys.argv[5])
	else: 
		mode = 0 #Random vs most frequent initialization 
	#print('Min accuracy:',min_acc)
	dataset = retrieve_data(sourcefile,N)
	#print('Pearson product-moment correlation coefficients:\n',correlation(dataset,0,1),'\n',
	#	correlation(dataset,1,2),'\n',correlation(dataset,2,3),'\n',correlation(dataset,1,3),'\n' )
	print(datetime.now())
	driver_cluster_full_address(dataset,N,kp,min_acc,mode)
	print(datetime.now())

driver()
