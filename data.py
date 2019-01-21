import pymongo,csv,math
from bson.json_util import loads

class DataSource:
	def __init__(self):
		pass

	def IPdatabase(self):
		client = pymongo.MongoClient('localhost', 27017)
		db = client.IPdatabase2 
		return db

	def IPdatabase2(self):
		client = pymongo.MongoClient('localhost', 27017)
		db = client.IPdatabase
		return db

	def getrawdata(self,collection='data_aws_us_west_1',appendname=False, collindex=None):
		db = self.IPdatabase()
		cursor = db[collection].find().sort('date',pymongo.ASCENDING)
		if appendname:
			if collindex is not None:
				return [ [doc['b1'],doc['b2'],doc['b3'],doc['b4'],collindex] for doc in cursor]	
			else: 
				return [ [doc['b1'],doc['b2'],doc['b3'],doc['b4'],collection] for doc in cursor]	
		return [ [doc['b1'],doc['b2'],doc['b3'],doc['b4']] for doc in cursor]

	def getbaredata(self,collection):
		db = self.IPdatabase()
		cursor = db[collection].find().sort('date',pymongo.ASCENDING)#.limit(4000)
		return list(cursor)

	def getbareIPaddresses(self,collection,byte):
		db = self.IPdatabase()
		cursor = db[collection].find({},{"b"+str(byte):1}).sort('date',pymongo.ASCENDING)#.limit(4000)
		return [rec['b'+str(byte)] for rec in list(cursor)]

	def getbareIPaddresses016(self,collection):
		db = self.IPdatabase()
		cursor = db[collection].find({},{"b1":1,"b2":1}).sort('date',pymongo.ASCENDING)#.limit(4000)
		return [str(rec['b1'])+str(rec['b2']) for rec in list(cursor)]

	def getthirdwithdate(self,collection='data_aws_ireland'):
		db = self.IPdatabase()
		cursor = db[collection].aggregate([
			{'$group': {'_id': {'address': '$b3' }, 'date': {'$push': '$date' } , 'count': {'$sum': 1} } }
		])
		return list(cursor) 

	def datasets(self):
		db = self.IPdatabase()
		sets = db.collection_names()
		res =  sorted(list(sets))
		return [r for r in res if r.find('SYN') == -1]

	def unique_values(self,collection,upto):
		db = self.IPdatabase()
		cursor = db[collection].find()
		IPset = set()
		for row in cursor: 
			strip = ''
			i = 0 
			for i in range(1,upto):
				strip += str(row['b'+str(i)])+'.'
			strip+=str(row['b'+str(i+1)])
			#strip = str(row['b1'])+'.'+str(row['b2'])+'.'+str(row['b3'])+'.'+str(row['b4'])
			IPset.add(strip)
		return IPset 

	def dataset_size(self,collection):
		db = self.IPdatabase()
		cursor = db[collection].count()
		return cursor

	def unique_values2(self,collection,upto,limit=0):
		db = self.IPdatabase()
		projected = dict()
		for i in range(1,upto+1):
			projected[str(i)] = '$b'+str(i)
		if limit > 0:
			cursor = db[collection].aggregate(
					   [
					   	  {'$sort': {'date': -1} },
					   	  { '$limit' : limit },

					      {
					        '$group' : {
					           '_id' : projected,
					           'count': { "$sum": 1 }
					        }
					      }, 

					   ]
					)
		else: 
			cursor = db[collection].aggregate(
					   [
					   	  {'$sort': {'date': -1} },

					      {
					        '$group' : {
					           '_id' : projected,
					           'count': { "$sum": 1 }
					        }
					      }, 

					   ]
					)
		return cursor

	def string_unique_values(self,region,limit):
		DS = DataSource()
		db = DS.IPdatabase()
		cursor = db[region].aggregate(
						   [
						   	  {'$sort': {'date': -1} },
						   	  { '$limit' : limit },
						   	  {'$project': {
									 'address': { '$concat': [{'$toString': '$b1'},'.',{'$toString': '$b2'},'.',{'$toString': '$b3'}] }
								}
								},
						      {
						        '$group' : {
						           '_id' : {'address': '$address'},
						           'count': { "$sum": 1 }
						        }
						      }, 

						   ]
						)
		values = list(cursor)
		return sorted(values,key=lambda x:x['count'],reverse=True)	

	def unique_values_per_day(self,collection,upto):
		db = self.IPdatabase()
		projected = []
		for i in range(1,upto+1):
			projected.append('$b'+str(i))
		cursor = db[collection].aggregate(
				   [
				   	  {'$sort': {'date': -1} },
				   	  {'$project':
				         {
				          'yearMonthDayUTC': { '$dateToString': { 'format': "%Y-%m-%d", 'date': "$date" } },
				          'address': projected,
				         }
				      },
				      {
				        '$group' : {
				           '_id' :  {'date': '$yearMonthDayUTC', 'address': '$address'},
				           'count': { "$sum": 1 }
				        }
				      }, 

				   ]
				)
		return cursor

	def unique_values_per_day_foraddress(self,collection,address,upto):
		db = self.IPdatabase()
		projected = []
		for i in range(1,upto+1):
			projected.append('$b'+str(i))
		cursor = db[collection].aggregate(
				   [
				   	  {'$sort': {'date': -1} },
				   	  {'$project':
				         {
				          'yearMonthDayHourUTC': { '$dateToString': { 'format': "%Y-%m-%d-%H", 'date': "$date" } },
				          'address': projected,
				         }
				      },
				      { '$match' : { 'address' : address } },
				      {
				        '$group' : {
				           '_id' :  {'date': '$yearMonthDayHourUTC', 'address': '$address'},
				           'count': { "$sum": 1 }
				        }
				      }, 

				   ]
				)
		return cursor

	def value_frequencies(self,collection,upto):
		db = self.IPdatabase()
		values = self.unique_values2(collection,upto)
		count = self.dataset_size(collection)
		freqs = [] 
		for value in values:
			IP = str(value['_id']['1'])
			for i in range(2,upto+1):
				IP += '.'+str(value['_id'][str(i)])
			freq = value['count'] 
			relative_freq = round(freq/count,5)
			freqs.append((IP,relative_freq,freq))
		return freqs #sorted(freqs,key=lambda x:x[1],reverse=True)


	def rel_freq(self,collection,byte):
		db = self.IPdatabase()
		count = int(db[collection].count())
		cursor = db[collection].aggregate([ { '$group': {'_id': "$b"+str(byte), 'count':{'$sum': 1}}}])
		res = []
		i = 0
		for rec in cursor: 
			p = rec['count']/count
			res.append((rec['_id'], p, math.log(p), rec['count']))
			i+=1
		res = sorted(res, key=lambda x: x[1], reverse=True)

		res1 = round(-sum([p[1]*p[2] for p in res]),3), round(math.log(count),3), i

		cursor = db[collection].aggregate([ { '$group': {'_id': {'1':"$b1",'2':"$b2",'3':"$b3" }, 'count':{'$sum': 1}}}]) 

		res = []
		i = 0 
		for rec in cursor: 
			p = rec['count']/count
			res.append((rec['_id'], p, math.log(p)))
			i+=1 
		res = sorted(res, key=lambda x: x[1], reverse=True)
		res2 = round(-sum([p[1]*p[2] for p in res]),3), round(math.log(i),3), i
		return res1, res2

	def convert_all_to_int(self):
		db = self.IPdatabase()
		db2 = self.IPdatabase2()
		datasets = self.datasets()
		for dataset in datasets: 
			cursor = db[dataset].find()
			coll = db2[dataset]
			for row in cursor:
				for i in range(1,5): 
					row['b'+str(i)] = int(row['b'+str(i)])
				coll.insert_one(row)


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






