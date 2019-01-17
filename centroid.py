#centroid-based clustering
from data import DataSource
from sklearn.metrics import jaccard_similarity_score 
from scipy.spatial.distance import directed_hausdorff
from math import ceil,floor
import random,copy
import numpy as np
import statistics as stats 
import matplotlib.pyplot as plt

class CentroidCluster:
	def __init__(self):
		pass 
	def load_data(self,region,N,limit=0):
		DS = DataSource()
		db = DS.IPdatabase()
		maxlimit = db[region].count()
		if (limit <= 0) or (limit > maxlimit): 
			limit = maxlimit

		cursor = db[region].aggregate(
					   [
					   	  {'$sort': {'date': 1} },
					   	  {'$limit': limit},
					   	  {'$project': {
					   	  		 #'address': { '$concat': [{'$toString': '$subnet1'},'.',{'$toString': '$subnet2'},'.',{'$toString': '$subnet3'},'.',{'$toString': '$subnet4'}] }
								 'address': { '$concat': [{'$toString': '$subnet1'},'.',{'$toString': '$subnet2'},'.',{'$toString': '$subnet3'}] }
								 #'address': { '$concat': [{'$toString': '$subnet1'},'.',{'$toString': '$subnet2'}] }
							}
							},
					   ]
					)	
		seq = list(cursor)
		seq = [v['address'] for v in seq] 
		return np.array_split(seq,len(seq)//N)

	def load_unique_data_with_frequencies(self,region,N,limit=0):
		DS = DataSource()
		db = DS.IPdatabase()
		maxlimit = db[region].count()
		if (limit <= 0) or (limit > maxlimit): 
			limit = maxlimit
		cursor = db[region].aggregate(
					   [
					   	  {'$sort': {'date': 1} },
					   	  {'$limit': limit},
					   	  {'$project': {
					   	  		 #'address': { '$concat': [{'$toString': '$subnet1'},'.',{'$toString': '$subnet2'},'.',{'$toString': '$subnet3'},'.',{'$toString': '$subnet4'}] }
								 'address': { '$concat': [{'$toString': '$subnet1'},'.',{'$toString': '$subnet2'},'.',{'$toString': '$subnet3'}] }
								 #'address': { '$concat': [{'$toString': '$subnet1'},'.',{'$toString': '$subnet2'}] }
							}
							},
						{
							'$group' : {
					           '_id' : {'address': '$address'},
					           'count': { "$sum": 1 }
					        }
						}
					   ]
					)	
		seq = list(cursor)
		seq = [(v['_id']['address'],v['count']) for v in seq] 
		seq = sorted(seq,key=lambda x:x[1],reverse=True)
		return [s[0] for s in seq]

	def unique_combinations(self,X):
		XC = copy.deepcopy(X)
		XC = np.concatenate(XC, axis=None)
		U  = np.unique(XC,axis=0) 
		return list(U) 
		


	def random_pick(self,data,indexes,M):
		selected = random.sample(indexes,k=M)
		cluster = selected #[data[s] for s in selected] 
		#Remove the selected ones for the next iteration
		for s in selected:
			indexes.remove(s)
		return cluster,indexes

	def seed_clusters(self,data,K):
		#Collect all indexes of data points so we can
		#randomly choose them iteratively later on.
		indexes = [i for i in range(len(data))]

		#Number of vectors in each cluster
		M = len(data)//K

		#Storage for clusters
		C = []

		for i in range(K):
			#Choose the cluster
			cluster,indexes = self.random_pick(data,indexes,M)
			C.append(cluster)

		return C 

	def numeric_vals(self,l): 
		return [tuple(map(int,v.split('.'))) for v in l]

	def numeric_vals2(self,l): 
		return [int(v.replace('.','')) for v in l]

	def hausdorff(self,A,B):
		return (max([min([abs(x-b) for b in B]) for x in A])+
			   max([min([abs(x-b) for b in A]) for x in B]))

	def point_distance(self,X,point,cluster):
		dist = 0 
		#Convert the point into numeric tuple
		p = self.numeric_vals2(X[point])
		#For each other point
		for other in cluster:
			#Convert the other point into numeric tuple
			o = self.numeric_vals2(X[other])
			#Compute the distance
			d= self.hausdorff(p,o)
			#Sum the distances with all other points
			dist += d
		return dist

	def set_prototypes(self,X,clusters):
		#For each cluster
		for i in range(len(clusters)):
			#Store distance of each point to all other points
			distances = []
			#for each point
			for point in clusters[i] : 
				dist = self.point_distance(X,point,clusters[i])
				#Keep point's sum of distances for later
				distances.append((point,dist))
			#Sort ascending order, get the minimum index
			prototype = sorted(distances,key=lambda x:x[1])[0][0]
			#Find its place in the cluster.
			pindex = clusters[i].index(prototype)
			#Set first index in cluster to prototype 
			clusters[i][0],clusters[i][pindex] = clusters[i][pindex],clusters[i][0] 
		#Return clusters with all first indexes set to prototypes 
		return clusters

	def t_matrix(self,clusters):
		l = len(clusters)
		T = [[] for i in range(l)]

		for i in range(l):
			K = clusters[i]
			for j in range(l): 
				L = clusters[j]
				S = sum([1 for v in K if (v+1) in L]) 
				if S > 0.0: 
					T[i].append((j,round(S/len(K),3)))
		return T 

	#Given a point, which cluster prototype is closest? 
	def best_cluster(self,X,clusters,P): 
		distances = []
		point = self.numeric_vals2(X[P]) 
		for i in range(len(clusters)):
			prototype = self.numeric_vals2(X[clusters[i][0]])
			s = self.hausdorff(point,prototype)
			distances.append((s,i))

		#Sort the distances from smallest to largest
		distances.sort(key=lambda x: x[0])
		#Find the ties
		distances = [dist for dist in distances if dist[0]==distances[0][0]]
		#Favor the one that has this point as a prototype. 
		m = -1 
		for dist in distances:
			i = dist[1]
			#Is this cluster's prototype the same as the data point P?
			if clusters[i] == P: 
				m = i 
				break 
		#Otherwise, favor the minimum index
		if m == -1:
			m = min(distances, key=lambda x: x[1])[1]
		return m

	#Find the best cluster for a single new observation 
	def which_cluster(self,X,clusters,point):
		distances = []
		point = self.numeric_vals2(point) 
		for i in range(len(clusters)):
			prototype = self.numeric_vals2(X[clusters[i][0]])
			s = self.hausdorff(point,prototype)
			distances.append((s,i))

		#Sort the distances from smallest to largest
		distances.sort(key=lambda x: x[0])
		#Find the ties
		distances = [dist for dist in distances if dist[0]==distances[0][0]]
		return min(distances, key=lambda x: x[1])[1]


	def recluster(self,X,clusters):
		#Collect only the prototypes, then add the points according to the distance.
		new_clusters = [ [cluster[0]] for cluster in clusters ]

		for p in range(len(X)): 
			i = self.best_cluster(X,clusters,p)
			if new_clusters[i][0] != p: 
				new_clusters[i].append(p)
			
		return new_clusters


	def convergence_criteria(self,X,clusters,xclusters):
		SC = 0
		for cluster in clusters: 
			SC+= self.point_distance(X,cluster[0],cluster)
		SX = 0 
		for cluster in xclusters: 
			SX+= self.point_distance(X,cluster[0],cluster)

		#print(SX,SC,abs(SX-SC))
		return (SX==SC)

	def kmeans(self,X,clusters): 
		converged = False 
		xclusters = copy.deepcopy(clusters)
		i = 1 
		print('',end='')
		while not converged: 
			new_clusters = self.recluster(X,clusters)
			clusters = self.set_prototypes(X,new_clusters)
			converged = self.convergence_criteria(X,clusters,xclusters)
			xclusters = copy.deepcopy(clusters) 
			print('\r', end='')
			print('K-Means iteration', i, end="", flush=True)

			i+=1 
		print('\nDone')
		return clusters

	#X is a set of sets of strings, where each set is of size N. 
	#K is the number of total clusters.
	def hausdorffcluster(self,X,K,N):
		clusters = self.seed_clusters(X,K) 
		print('Clusters seeds created.')
		clusters = self.set_prototypes(X,clusters)
		print('Prototypes computed.')
		
		clusters = self.kmeans(X,clusters)
		T = self.t_matrix(clusters)
		print('Trans. matrix computed.')
		
		return clusters,T



	'''
	An implementation of the attacker algorithm
	X_train: Training sequence
	X_test: Test sequence 
	'''
	def simulate_attack(self,X_train,X_test,T,clusters,region,unique): 
		#How many of the total attack attempts did not generate any correct prediction (failed)?
		failed = 0 
		#How many IP addresses tried in each attack attempt?
		counts = [] 
		#How many IP addresses predicted in each attack attempt?
		succeeded = []
		#How many total IP addresses generated by the target server?
		tried = 0 
		#How many attack iterations? 
		N = len(X_train[0])
		attack_attempts = 0 

		#First observation 
		Q = [len(X_train)-1]
		#Find cluster for observation 
		observed_cluster = self.best_cluster(X_train,clusters,Q[-1])
		print('',end='')
		for i in range(len(X_test)):
			#Set the current number of IP addresses tried to zero. 
			count = 0 
			
			#Target server's choice
			choice = X_test[i]

			#Which cluster has the current choice of target network?
			row_index = observed_cluster 

			#Which row in T corresponds to the observed cluster?
			row = T[row_index]
			#Sort the row by transition prob.
			row.sort(key=lambda x: x[1],reverse=True)	

			#Form the prediction list: 
			cluster_points = [] 
			for cluster in row: 
				cluster_points += list(np.concatenate([list(X_train[p]) for p in clusters[cluster[0]]], axis=None))

			correct_predictions = 0 

			choice = set(choice)
			checked = [] 
			for IP in cluster_points:
				if IP in checked: 
					continue  
				count+=1 
				if IP in choice: 
					correct_predictions+=1 
				if correct_predictions == len(choice):
					break 
				checked.append(IP)

			counts.append(count)
			succeeded.append(correct_predictions)			

			#Now learn the recent point
			X_train.append(choice)
			#Set n to the index of the recent point
			n = len(X_train)-1
			#Find and assign its cluster
			c = self.which_cluster(X_train,clusters,choice)
			#Add the point to the best cluster
			clusters[c].append(n)
			# #Recompute clusters 
			# clusters = self.kmeans(X_train,clusters)
			#Recompute T
			T = self.t_matrix(clusters)
			#Record observation
			Q.append(n)
			#Set cluster for observation 
			observed_cluster = c

			print('\r', end='', flush=True)
			print('Attack progress: ',end="") 
			s = "{:.2%}".format(round((attack_attempts/len(X_test)),2))
			print(s, end="")

			attack_attempts+=1 

		U = len(unique) 

		f = open('summary_'+region+'.txt','w')

		print('Unique IP addresses,',len(unique),file=f)
		print('K,',len(clusters),file=f)
		print('Cluster-based attack',file=f)
		print('Trials Mean,', round(stats.mean(counts),3), ',Trials Median,',round(stats.median(counts),3), ',Trials Minimum,', min(counts), ',Trials Maximum,', max(counts),file=f)
		print('Attack iterations,', attack_attempts, file=f)
		print('Accuracy Mean,', round(stats.mean(succeeded),3), ',Accuracy Median,',round(stats.median(succeeded),3), ',Accuracy Minimum,', min(succeeded), ',Accuracy Maximum,', max(succeeded),file=f)
		print('Accuracy:', round(sum(succeeded)/(len(succeeded)*N),6), file=f )	
		#plt.scatter([i for i in range(len(counts))], counts, marker="+")
		f.close()

		print('\nDone simulating cluster attacker.')

		return failed, counts 

	def simulate_random_attack(self,X_train,X_test,region,unique,maxtrials=1000): 
		#How many of the total attack attempts did not generate any correct prediction (failed)?
		failed = 0 
		#How many IP addresses tried in each attack attempt?
		counts = [] 
		#How many IP addresses predicted in each attack attempt?
		succeeded = []
		#How many total IP addresses generated by the target server?
		tried = 0 
		#How many attack iterations? 
		N = len(X_train[0])
		attack_attempts = 0 
		maxtrials = len(unique)-1
		print('',end='')
		for i in range(len(X_test)):
			#Set the current number of IP addresses tried to zero. 
			count = 0 
			
			#Target server's choice
			choice = X_test[i]
			
			#Form the prediction list: 
			cluster_points = list(random.sample(unique,k=maxtrials))

			correct_predictions = 0 

			#Form the prediction and choice sets: 
			choice = set(choice)
			checked = [] 
			for IP in cluster_points:
				if IP in checked: 
					continue  
				count+=1 
				if IP in choice: 
					correct_predictions+=1 
				if correct_predictions == len(choice):
					break 
				checked.append(IP)

			counts.append(count)
			succeeded.append(correct_predictions)	

			#Now learn the recent point
			X_train.append(choice)
			print('\r', end='', flush=True)
			print('Attack progress: ',end="") 
			s = "{:.2%}".format(round((attack_attempts/len(X_test)),2))
			print(s, end="")
			attack_attempts+=1 

		U = len(unique) 

		f = open('summary_'+region+'.txt','a+')
		print('Random attack',file=f)
		print('Trials Mean,', round(stats.mean(counts),3), 'Trials Median,',round(stats.median(counts),3), 'Trials Minimum,', min(counts), 'Trials Maximum,', max(counts),file=f)
		print('Attack iterations,', attack_attempts, file=f)
		print('Accuracy Mean,', round(stats.mean(succeeded),3), 'Accuracy Median,',round(stats.median(succeeded),3), 'Accuracy Minimum,', min(succeeded), 'Accuracy Maximum,', max(succeeded),file=f)
		print('Accuracy:', round(sum(succeeded)/(len(succeeded)*N),6), file=f )	
		#print(sorted(counts),file=f)
		#print(succeeded,file=f)

		#plt.scatter([i for i in range(len(counts))], counts, marker="v")

		f.close()
		print('\nDone simulating random attacker.')

		return failed, counts

	def simulate_frequency_attack(self,X_train,X_test,region,unique,maxtrials=1000): 
		#How many of the total attack attempts did not generate any correct prediction (failed)?
		failed = 0 
		#How many IP addresses tried in each attack attempt?
		counts = [] 
		#How many IP addresses predicted in each attack attempt?
		succeeded = []
		#How many total IP addresses generated by the target server?
		tried = 0 
		#How many attack iterations? 
		N = len(X_train[0])
		attack_attempts = 0 
		print('',end='')
		maxtrials = len(unique)
		for i in range(len(X_test)):
			#Set the current number of IP addresses tried to zero. 
			count = 0 
			
			#Target server's choice
			choice = X_test[i]
			
			#Form the prediction list: 
			cluster_points = unique[:maxtrials]

			correct_predictions = 0 

			#Form the prediction and choice sets: 
			choice = set(choice)
			checked = [] 
			for IP in cluster_points:
				if IP in checked: 
					continue  
				count+=1 
				if IP in choice: 
					correct_predictions+=1 
				if correct_predictions == len(choice):
					break 
				checked.append(IP)

			counts.append(count)
			succeeded.append(correct_predictions)	

			#Now learn the recent point
			X_train.append(choice)
			print('\r', end='', flush=True)
			print('Attack progress: ',end="") 
			s = "{:.2%}".format(round((attack_attempts/len(X_test)),2))
			print(s, end="")
			attack_attempts+=1 

		U = len(unique) 

		f = open('summary_'+region+'.txt','a+')
		print('Frequency attack',file=f)
		print('Trials Mean,', round(stats.mean(counts),3), 'Trials Median,',round(stats.median(counts),3), 'Trials Minimum,', min(counts), 'Trials Maximum,', max(counts),file=f)
		print('Attack iterations,', attack_attempts, file=f)
		print('Accuracy Mean,', round(stats.mean(succeeded),3), 'Accuracy Median,',round(stats.median(succeeded),3), 'Accuracy Minimum,', min(succeeded), 'Accuracy Maximum,', max(succeeded),file=f)
		print('Accuracy:', round(sum(succeeded)/(len(succeeded)*N),6), file=f )	
		#print(sorted(counts),file=f)
		#print(succeeded,file=f)

		#plt.scatter([i for i in range(len(counts))], counts, marker="*")

		# plt.legend(['Cluster','Random','Frequency'],loc=1, prop={'size': 8})
		# plt.ylabel('Number of guesses to predict N IP addresses')
		# plt.xlabel('Attack iteration')
		# plt.savefig('results_'+region+'.pdf',format='pdf')
		f.close()
		print('\nDone simulating random attacker.')

		return failed, counts

	



		 