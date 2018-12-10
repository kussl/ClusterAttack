import pymongo,csv
from bson.json_util import loads

class DataSource:
	def __init__(self):
		pass

	def IPdatabase(self):
		client = pymongo.MongoClient('localhost', 27017)
		db = client.IPdatabase 
		return db

	def getrawdata(self,collection='data_aws_us_west_1',appendname=False, collindex=None):
		db = self.IPdatabase()
		cursor = db[collection].find()
		if appendname:
			if collindex is not None:
				return [ [doc['subnet1'],doc['subnet2'],doc['subnet3'],doc['subnet4'],collindex] for doc in cursor]	
			else: 
				return [ [doc['subnet1'],doc['subnet2'],doc['subnet3'],doc['subnet4'],collection] for doc in cursor]	
		return [ [doc['subnet1'],doc['subnet2'],doc['subnet3'],doc['subnet4']] for doc in cursor]

	def getthirdwithdate(self,collection='data_aws_ireland'):
		db = self.IPdatabase()
		cursor = db[collection].aggregate([
			{'$group': {'_id': {'address': '$subnet3' }, 'date': {'$push': '$date' } , 'count': {'$sum': 1} } }
		])
		return list(cursor) 

	def datasets(self):
		db = self.IPdatabase()
		sets = db.collection_names()
		return list(sets)
	

class DataDriver:
	def __init__(self):
		pass

	def dumpalldatatocsv(self):
		DS = DataSource()
		db = DS.IPdatabase()
		collections = [('data_aws_ap_northeast_1',0), ('data_aws_ca_central_1',1), 
		('data_aws_eu-west-3',2), ('data_aws_ireland',3), ('data_aws_sa_east_1',4), ('data_aws_us_west_1',5),('data_gce_us',6)]

		f = open('everything.csv','w')
		writer = csv.writer(f)
		data = [] 
		for collection,index in collections: 
			records = DS.getrawdata(collection=collection,appendname=True,collindex=index)
			for row in records: 
				writer.writerow(row)
				data.append(row)

		f.close()
		return data 

	def dumpalldata(self):
		DS = DataSource()
		db = DS.IPdatabase()
		collections = [('data_aws_ap_northeast_1',0), ('data_aws_ca_central_1',1), 
		('data_aws_eu-west-3',2), ('data_aws_ireland',3), ('data_aws_sa_east_1',4), ('data_aws_us_west_1',5),('data_gce_us',6)]

		data = [] 
		for collection,index in collections: 
			records = DS.getrawdata(collection=collection,appendname=True,collindex=index)
			for row in records: 
				data.append(row)
		return data 






