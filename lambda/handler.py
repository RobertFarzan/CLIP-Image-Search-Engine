import json
import boto3
import pickle
from transformers import CLIPProcessor, TFCLIPModel
from datetime import datetime

TABLE_NAME = "CLIPEmbeddingsTable"
s3_resource = boto3.resource('s3', region_name='us-east-1')
dynamodb_resource = boto3.resource('dynamodb', region_name='us-east-1')
dynamodb_table = dynamodb_resource.Table(TABLE_NAME)

model = TFCLIPModel.from_pretrained(os.path.join(ASSETS_PATH, 'clip-vit-base-patch32'))
processor = CLIPProcessor.from_pretrained(os.path.join(ASSETS_PATH, 'clip-vit-base-patch32'))

def lambda_handler(event, context):
	
