import random,math
import numpy as np 
from sklearn.model_selection import train_test_split
from distance import Distance

class CrossValidator:
	def __init__(self):
		pass
	'''
	This function will produce the transition matrix for the Markov process
	as defined in the manuscript. Each T[i] will have the number of transitions
	from a cluster C_i to a cluster C_j where the transition is defined as having
	values A_k in C_i and A_k+1 in C_j where 1<=k<m. 
	The matrix will be represented by a list. 
	'''
	def transition_matrix(self,clusters): 
		#Number of clusters
		l = len(clusters)

		#Create a zero matrix with lists. I don't think numpy would be of any benefit. 
		T = [ [0 for i in range(l)] for j in range(l) ]

		#Loop over each cluster, then over each assignment.
		for i in range(l):
			#Collect all k's in each assignment. 
			K = [assignment.k for assignment in clusters[i].assignments]

			#Now compare with all clusters, including self. 
			for j in range(l): 
				L = [assignment.k for assignment in clusters[j].assignments]

				#Assign T[i,j] the number of values v in K with a v+1 in L.
				S = sum([1 for v in K if (v+1) in L]) 
				T[i][j] = round(S/len(K),3) 

		return T 

	'''
	Compute accuracy of prediction by finding the intersection of
	the target server's choice with the predicted addresses. 
	We will consider the target server's choice of addresses as one
	set A and the set of ALL addresses in the prediction as B. 
	|A intersect B| is the accuracy rate. 
	To make it work, we will convert the leading 16 bits to one string,
	which will be kept in a set. 
	'''
	def prediction_accuracy(self,choice,predictions):
		A = {str(IP[0])+str(IP[1]) for IP in choice.addresses.tolist()}
		B = set()
		for p in predictions:
			B|={str(IP[0])+str(IP[1]) for IP in p.addresses.tolist()}
		return len(A & B)


	'''
	Cross-validate the results of a cluster. 
	Except assigning an accuracy value to each cluster, should not modify anything else.
	'''
	def cross_validate(self,X_test,clusters,T,N):
		DT = Distance()
		#Frequency of the number of correct predictions wrt N:
		F = [0]*(N+1)
		#Return the prediction set
		P = [] 

		#Roll through the test set. 
		for i in range(len(X_test)-1):
			#Last observation of the attacker:
			observed = X_test[i]
			#Which cluster does it belong to?
			observed_cluster = DT.closest_cluster(observed,clusters)
			#Target server's choice
			choice = X_test[i+1]
			#Which cluster does it belong to?
			choice_cluster = DT.closest_cluster(choice,clusters)
			#What do clusters say about the next choice of server?
			#Take TWO clusters as the most likely.
			row = T[observed_cluster].index(max(T[observed_cluster]))
			predictions = [] 
			predictions += clusters[row].assignments
			#predictions.append(clusters[row].prototype)
			#How good was the prediction?
			accuracy = self.prediction_accuracy(choice,predictions)
			clusters[row].accuracy = round((clusters[row].accuracy+accuracy)/2,3)
			F[accuracy]+=1 
			P.append(predictions)

		#Number of total predictions
		n = len(X_test)
		return clusters,[round(f/n,3) for f in F],P


	'''
	The same as prediction_accuracy, but for the entire IP address.
	'''
	def prediction_accuracy_full(self,choice,predictions):
		A = {str(IP[0])+str(IP[1])+str(IP[2]) for IP in choice.addresses.tolist()}
		B = set()
		for p in predictions:
			if len(p.full_addresses) > 0:
				addresses = p.full_addresses
			else:
				addresses = p.addresses
			B|={str(IP[0])+str(IP[1])+str(IP[2]) for IP in list(addresses)}
		if len(B) == 0:
			raise Exception('Empty predictions', p.full_addresses)
		return len(A & B)

	'''
	Cross-validate a FULL IP address using three clusters. 
	Here, we will pick an observed and a choice address, A. Then, use three 
	matrixes T_1, T_2, and T_3 to pick the bits 0--15, then bits 9--23, and finally
	bits 16--32. 
	Using all the matrixes, find three prototypes as predictions. Then, 
	intersect all the prototypes in a single set B, and measure the intersection
	of A and B as the quality of prediction.
	The set B must include values for the right bits. For example, if 
	from T_1, we conclude 54120 as the first 16 bits and from T_2, we concluded
	12066 for the bits 8--24, then we can construct the values for 0--24. 
	Here, clusters is a set of cluster sets and T is a set of matrixes. 
	clusters_set is assumed to include clusters for bytes (0,1), (1,2), and 2,3. 
	'''
	def hierarchical_cross_validate(self,X_tests,clusters_set,T_set,N):
		DT = Distance()
		#Frequency of the number of correct predictions wrt N:
		F = [0]*(N+1)
		#Roll through the test set. 
		X_test = X_tests[0] 
		#Avg. number of assignments used for prediction.
		#This will vary when using entire clusters.
		assignments_used = 0 

		for i in range(len(X_test)-1):
			observed_clusters = [] 
			choice_clusters = [] 
			predictions = [] 
			#Last observation of the attacker:
			observed = X_test[i]
			#Target server's choice
			choice = X_test[i+1]
			for j in range(2):
				clusters = clusters_set[j]
				T = T_set[j]
				#Last observation of the attacker:
				observed = X_test[i]
				#Which cluster does it belong to?
				observed_cluster = DT.closest_cluster(observed,clusters)
				#Target server's choice
				choice = X_test[i+1]
				#Which cluster does it belong to?
				choice_cluster = DT.closest_cluster(choice,clusters)
				#What do clusters say about the next choice of server?
				#Take TWO clusters as the most likely.
				row = T[observed_cluster].index(max(T[observed_cluster]))
				#candidates.append(clusters[row].assignments)
				k = len(clusters[row].assignments)
				if k > 40:
					k = 40 
				predictions += clusters[row].assignments[:k]
				#predictions += random.sample(clusters[row].assignments,k)
				#predictions += [clusters[row].prototype]

			if len(predictions) < 3: 
				raise Exception('Could not find three clusters for the choice. This must alsways be three!')		

			#How good was the prediction?
			# print('Choice:', choice)
			# print('Obs clusters:', observed_clusters, j)
			# print('Pred:', [p.addresses for p in candidates])
			assignments_used = (assignments_used +len(predictions))/2
			accuracy = self.prediction_accuracy_full(choice,predictions)
			clusters[row].accuracy = round((clusters[row].accuracy+accuracy)/2,3)
			F[accuracy]+=1 

		#Number of total predictions
		n = len(X_test)
		return clusters,[round(f/n,3) for f in F],assignments_used