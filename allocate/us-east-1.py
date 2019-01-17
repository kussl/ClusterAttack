import boto3
from bson.json_util import dumps

ec2 = boto3.client('ec2', region_name='us-east-1')

out = open('/home/almohri/experiments/US-EAST-1', 'a+')
for i in range(0,5):
   try: 
      allocation = ec2.allocate_address(Domain='standard')
      out.write(dumps(allocation))
      out.write('\n')
   except: 
      pass

out.close()

addresses_dict = ec2.describe_addresses()
for eip_dict in addresses_dict['Addresses']:
   try: 
      response = ec2.release_address(AllocationId=eip_dict['AllocationId'])
   except:
      pass 
