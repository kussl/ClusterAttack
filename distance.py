from ds import Dataset, Assignment, Prototype, Cluster

class Distance:
	#d, H, and distance compute the Hausdorff distance between two addresses A and B. 
	def d(self,x, B): 
		diff=[]
		for v in B:
			diff.append(abs(v-x))
		return min(diff)

	def H(self,A,B):
		diff=[]
		for v in A:
			diff.append(self.d(v,B))
		return max(diff)

	#Compute the distance between two sets. 
	def distance(self,A,B):
		return self.H(A,B)+self.H(B,A)

	#Compute the sum of distance of an assignment P 
	#with all assignments in a cluster.
	def dist_sum(self,P,cluster):
		distances = []
		for assignment in cluster.assignments: 
			distances.append(self.distance(assignment.values,P.values))
		dist_sum = sum(distances)
		P.distance = dist_sum
		return dist_sum

	#Find the prototype assignment in a cluster.
	def minimize_dist(self,cluster):
		#All distance sums in a list.
		distance_sums = []
		#For each assignment, compute its distance sum and record it. 
		for i in range(len(cluster.assignments)): 
			distance_sums.append((self.dist_sum(cluster.assignments[i],cluster),cluster.assignments[i]))

		#Which one had the minimum value?
		P = min(distance_sums, key = lambda t: t[0])

		#Tag it as a prototype for its cluster. A pointer to the assignment is enough.
		cluster.prototype = P[1]
		return cluster

	#Find the closest cluster for an assignment. 
	def closest_cluster(self,P,clusters):
		distances = [] 
		for i in range(len(clusters)):
			s = self.distance(P.values, clusters[i].prototype.values)
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
			if clusters[i].prototype.k == P.k: 
				m = i 
				break 
		#Otherwise, favor the minimum index
		if m == -1:
			m = min(distances, key=lambda x: x[1])[1]
		return m


