
class ImproveKMeans:
	def __init__(self):
		pass 

	'''
	Split and check one cluster. 
	This one is used by split. 
	'''
	def split_one(self,cluster):
		pass

	'''
	Split clusters to improve the prediction accuracy. 
	Sort clusters by length. Take the top x% longest clusters
	and split them into two. Check if the overall prediction accuracy 
	improves after each split. Record the best split. 
	'''	
	def split(self,clusters):
		pass