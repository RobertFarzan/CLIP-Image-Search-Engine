import json
import boto3
import botocore
import pickle
import os
import numpy as np
from transformers import CLIPProcessor, TFCLIPModel
from datetime import datetime
from PIL import Image

LAMBDA_TASK_ROOT = os.environ.get('LAMBDA_TASK_ROOT', '/var/task')
TABLE_NAME = "CLIPEmbeddingsTable"
RESIZE = (256,256)

s3_resource = boto3.resource('s3', region_name='us-east-1')
dynamodb_resource = boto3.resource('dynamodb', region_name='us-east-1')
dynamodb_table = dynamodb_resource.Table(TABLE_NAME)

model = TFCLIPModel.from_pretrained(LAMBDA_TASK_ROOT)
processor = CLIPProcessor.from_pretrained(LAMBDA_TASK_ROOT)

def lambda_handler(event, context):
    
    for record in event.get('Records', []):
        # 1 LEER DATOS DESDE S3
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        obj = s3_resource.Object(bucket, key)
        
        element = obj.get()['Body']
        img = Image.open(element).resize(RESIZE)
        inputs = processor(images=img, return_tensors="tf")
        
        print(f'Getting image features for the image {key}')
        image_features = model.get_image_features(**inputs).numpy().squeeze()

        print(f'Image features successfully extracted for the image {key}')
        print(f'Normalizing image features for the image {key}')
        image_features = image_features / np.linalg.norm(image_features, ord=2)
        ts_now = datetime.now().isoformat(timespec='microseconds')
        
        try:
            dynamodb_table.put_item(
                                    Item={
                                        'type_pk': "image",
                                        'timestamp': ts_now,
                                        'image_id': key,
                                        'embedding': pickle.dumps(image_features)
                                    }
                                )
        except botocore.exceptions.ClientError as error:
            print('Error putting item:', error.response['Error']['Message'])

            return {
                    'statusCode': 500,
                    'body': json.dumps('Upload to DynamoDB failed')
            }
        else:
            print(f'Item with ID {key} uploaded successfully to DynamoDB [{ts_now}]')

    return {
        'statusCode': 200,
        'body': json.dumps(f'Successful upload of image {key} to DynamoDB')
    }
    
	
