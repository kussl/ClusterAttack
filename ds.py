import random
from scipy.spatial import distance
from numpy import average as avg
import numpy as np 
from threading import Thread
import random,copy,statistics
import uuid
import datetime

class Dataset:
  def __init__(self,assignments=[],N=5):
    #List of all assignments, each containing N addresses 
    self.assignments = assignments
    #Number of addresses in each assignment 
    self.N = N

  def assign(self,assignment): 
    if isinstance(assignment,Assignment):
      assignment = copy.deepcopy(assignment) 
      self.assignments.append(assignment)
    else:
      raise Exception('Assignment of unexpected type.') 

  def __getitem__(self, i):
    return self.assignments[i]
  def __len__(self):
    return len(self.assignments)
  # def __iter__(self):
  #   return self.assignments.__iter__()

class Assignment: 
  def __init__(self,addresses=[],k=0,full_addresses=[]):
    #Addresses of a single assignment. 
    self.addresses = copy.deepcopy(addresses) 
    self.full_addresses =  copy.deepcopy(full_addresses) 
    #The numberic value computed by the score function for each assignment. 
    self.values = self.score(addresses)
    #Order of assignment in time. 
    self.k = k
    #Is the assignment clustered?
    self.clusterid = None 
    self.clusterindex = None 
    self.cluster = None 
    #This is the sum of this assignment's distance to all other assignments
    #in its cluster.
    self.distance = None
    #When clustering bytes beyond the first two, we will need
    #the full address.  
    
    

  def __str__(self):
    return self.addresses.__str__()
  def score(self,X):
    return [ int(str(x[0])+str(x[1])) for x in X]

  def equiv(self,assignment): 
    return (assignment.k == self.k) 

  def similarvalues(self,assignment):
    for v in self.values:
      if v not in assignment.values:
        return False
    return True 
    # addresses = assignment.addresses
    # for addr in addresses: 
    #   found=False 
    #   for addr2 in self.addresses: 
    #     if addr.tolist() == addr2.tolist(): 
    #       found=True 
    #   if not found: 
    #     return False  
    # return True

  def similarity(self,assignment,upto=2): 
    addresses = assignment.addresses
    score = 0 
    for addr in addresses: 
      s = 0 
      for addr2 in self.addresses: 
        if addr[:upto].tolist() == addr2[:upto].tolist(): 
          s+=1 
          break 
      if s >0: 
        score+=1
    return score 


class Prototype(Assignment):
  def __init__(self, assignment):
    #Sum of distance with all other members of the cluster.
    self.distance = 0 
    self.addresses = copy.deepcopy(assignment.addresses)
    self.values    = copy.deepcopy(assignment.values) 
    self.k         = assignment.k 
    self.clusterid = assignment.clusterid
    
    

class Cluster():
  def __init__(self,assignments=[],prototype=None,cid=0):
    #An identifier for the cluster.
    self.cid = str(uuid.uuid4())
    self.cid = self.cid[:len(self.cid)//2]
    self.assignments = []
    #The prototype assignment. 
    self.prototype   = prototype
    self.assign(assignments)
    #Squared error 
    self.se = 0 
    #Mean prediction error
    self.accuracy = 0 

  def assign(self, assignments):
    #Single assignment
    if isinstance(assignments, Assignment):
      nassignment = Assignment(k=assignments.k) 
      nassignment.addresses = copy.deepcopy(assignments.addresses)
      nassignment.full_addresses = copy.deepcopy(assignments.full_addresses)
      nassignment.values    = copy.deepcopy(assignments.values) 
      nassignment.clusterid = self.cid 
      nassignment.cluster = self
      self.assignments.append(nassignment)
    else: 
      for assignment in assignments: 
        nassignment = Assignment(k=assignment.k) 
        nassignment.addresses = copy.deepcopy(assignment.addresses) 
        nassignment.full_addresses = copy.deepcopy(assignment.full_addresses)
        nassignment.values    = copy.deepcopy(assignment.values) 
        nassignment.clusterid = self.cid 
        nassignment.cluster = self
        self.assignments.append(nassignment)

