import random
from scipy.spatial import distance
from numpy import average as avg
from crossval import CrossValidator  
import numpy as np 
from threading import Thread
import random,copy,statistics
import uuid,math,datetime
from math import fabs
from ds import Dataset, Assignment, Prototype, Cluster
from distance import Distance 
class KMeans:
	def __init__(self):
		pass

	#Compute all the prototypes for all clusters
	def prototypes(self, clusters):
		DT = Distance()
		for i in range(len(clusters)):
			clusters[i] = DT.minimize_dist(clusters[i])
		return clusters


	'''
	Given available indexes, pick a random data point 
	to form a cluster.
	M is the number of data points in each cluster.
	Indexes are the available ones to pick from. 
	'''
	def initializer_random_pick(self,data,indexes,M):
		selected = random.sample(indexes,k=M)
		cluster = Cluster(assignments=[data[s] for s in selected])
		#Remove the selected ones for the next iteration
		for s in selected:
			indexes.remove(s)
		return cluster,indexes

	'''
	Given all the data points, select the K most frequent points
	and regard them as the prototypes by creating K clusters of only
	single data points.

	1. Go through data and find the mode.   
	2. Find the index of the mode.
	3. Remove the index of ALL values that are identical to the mode. 
	4. Add the mode to the cluster and return the cluster and new indexes.

	'''
	def initializer_most_frequent(self,data,indexes,M):
		#Find the frequency of each assignment. 
		frequencies = []
		for i in indexes: 
			#frequencies.append( (i, data.assignments.count(data[i])) )
			freq = sum(data[i].similarvalues(P) for P in data)
			frequencies.append((i,freq))

		#Find the maximum frequency
		mf = max(frequencies, key=lambda x: x[1])

		#Remove all the data assignments are identical to mf. 
		assignment = data[mf[0]]
		while True: 
			#Find the next occurance of assignment: 
			index = data.assignments.index(assignment)
			if index == -1:
				break 
			try:
				indexes.remove(index)
			except Exception as e:
				break 

		return Cluster(assignments=[assignment]), indexes

	'''
	Initialize K clusters to be used later within the clustering loop.
	The initializers could be different: random or based on frequency. 
	'''
	def initialize_clusters(self,data,K,mode=0):
		if not isinstance(data,Dataset):
			raise Exception('Dataset of unexpected type.')

		#Collect all indexes of data points so we can
		#randomly choose them iteratively later on.
		indexes = [i for i in range(len(data))]

		#Number of vectors in each cluster
		M = len(data)//K

		#Storage for clusters
		C = []

		for i in range(K):
			#Choose the cluster
			if mode == 0: 
				cluster,indexes = self.initializer_random_pick(data,indexes,M)
			else: 
				cluster,indexes = self.initializer_most_frequent(data,indexes,M)
			C.append(cluster)
		#print(K,' initial clusters formed.', end=' ')			
		#Compute prototypes
		C = self.prototypes(C)
		return C

	'''
	This function determines if a clustering algorithm
	has reached convergence by examining the current cluster
	against the previous one (xcluster). 
	Returns true if converged. 
	'''
	def dist_convergence(self, clusters, xclusters):
		prototypes = [c.prototype for c in clusters]
		xprototypes = [c.prototype for c in xclusters]

		SC = 0
		for cluster in clusters: 
			SC+= cluster.prototype.distance
		SX = 0 
		for cluster in xclusters: 
			SX+= cluster.prototype.distance

		#print(SX,SC,abs(SX-SC))
		return (SX==SC), abs(SX-SC)


	'''
	Assign data points to their respective clusters
	by minimizing the distance with the prototypes. 
	'''
	def assign(self,data,clusters,K):
		DT = Distance()
		#For each data point, which cluster is closest?
		for P in data:
			m = DT.closest_cluster(P,clusters)
			clusters[m].assign(P) 

		#An empty cluster is NOT acceptable
		for i in range(len(clusters)):
			if len(clusters[i].assignments)== 0:
				raise Exception('Empty cluster found..')

		C = self.prototypes(clusters)

		return C


	'''Do all that is required to group
	the data into K clusters.
	It should converge to a point at which
	the cluster prototype values do not show
	significant difference in iterations.
	Assign according to nearest prototype. 
	This only performs ONE round of clustering. 
	'''
	def cluster_alg(self,data,K,clusters):
		converged = False 
		xclusters = copy.deepcopy(clusters)
		i=1
		while not converged:
			#Empty the cluster, leave the prototypes INSIDE 
			#the assignments list. The reason to do this is because
			#prototypes may be reassigned to other clusters and be left
			#out of their own clusters. 
			for i in range(len(clusters)):
				P = clusters[i].prototype
				clusters[i].assignments.clear()
				#clusters[i].assignments.append(P)
			i+=1
			clusters = self.assign(data,clusters,K)
			converged,sumdiff = self.dist_convergence(clusters, xclusters)
			xclusters = copy.deepcopy(clusters)
		return clusters

	'''
	A driver to run the clustering algorithm and then test the quality of the results. 
	This driver repeats the cluster until a minimum accruacy is assured.
	'''
	def cluster(self,X_train,X_test,K,N,M=20,min_acc=0.5,mode=0):
		#Repeat the execution until a minimum quality is acheived 
		score = 0
		pscore = 0 #Previous score 
		#Count the iterations
		i=0
		while score < min_acc:
			clusters = self.initialize_clusters(X_train,K,mode=mode)
			#print('init..')
			clusters = self.cluster_alg(X_train,K,clusters)
			#print('clustered...')
			PR = CrossValidator()
			T = PR.transition_matrix(clusters)
			clusters,F,P = PR.cross_validate(X_test,clusters,T,N)
			#Score means the sum of proportion of correct predictions more than N/2. 
			pscore = score 
			score = round(sum(F[math.ceil(N/2):]),3)
			print(score)
			i+=1 
			if fabs(score-pscore) < 0.05: 
				break 

		return clusters,score,T,F


