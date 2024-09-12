import pymongo
from pymongo import MongoClient
import os
os.environ["COHERE_API_KEY"]='1WDphHnJYzXRcm2EjDcvyqXRnKRG6n83XxX7LPFx'
os.environ['PINECONE_API_KEY']='ac2836be-5765-4e89-8d47-70ab0771e347'#'6dbebefb-e722-4241-8041-00f56ca935ca'
os.environ['PINECONE_ENV']='gcp-starter'
os.environ['QDRANT_API_KEY']='B2p7WN_t2TIpugdRgeZ-S5ApOPZ-VigWZZxhxDE036aBbATU_mpx1g'
os.environ['GOOGLE_API_KEY']='AIzaSyAUggwhrE0LoTBDWrfeU6kxQuxA0FP6eCk'
os.environ['APIFY_API_TOKEN']='apify_api_K90vlEcLcKMx43KED0DpKQuxz2cTUr2CXPtv'
os.environ['HUGGINGFACEHUB_API_TOKEN'] ='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'
client = MongoClient("mongodb+srv://shakthi:shakthi2002@cluster0.at8jmas.mongodb.net/")
db=client['chatHistory']
cht_info=db.cht_info
from datetime import datetime
import pytz  # For working with time zones
print('start Database file')

def insert_data(queryy,response):

  # Get the current date and time
  current_datetime = datetime.now()
  ist_timezone = pytz.timezone('Asia/Kolkata')

  # Convert the UTC time to IST
  current_time_ist = current_datetime.astimezone(ist_timezone)

  # Format the current date and time (optional)
  current_datetime = current_time_ist.strftime("%Y-%m-%d %H:%M:%S")
  cht_info.insert_one({'query': queryy,'response': response , 'time': current_datetime})


def get_chthistory():
  ist_timezone=pytz.timezone('Asia/Kolkata')
  cursor=cht_info.find().sort('time',pymongo.DESCENDING).limit(5)
  latest_cht_history=list()
  print('hii')
  for doc in cursor:
    qury=doc['query']
    resp=doc['response']
    tim=doc['time']
    latest_cht_history.append((qury,resp))

  return latest_cht_history
