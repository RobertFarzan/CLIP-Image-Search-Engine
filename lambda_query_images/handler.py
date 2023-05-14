import os
import json
import boto3
import botocore
import faiss
import numpy as np
from transformers import TFCLIPModel, CLIPTokenizer

LAMBDA_TASK_ROOT = os.environ.get('LAMBDA_TASK_ROOT', '/var/task')
BUCKET_METADATA_NAME = "clip.index.metadata.tfm.robert"
BUCKET_IMAGE_NAME = "clip.images.tfm.robert"

FAISS_INDEX = "faiss.index"
INDEX_TO_ID = "index_to_id.json"

s3 = boto3.client('s3', region_name='us-east-1')
s3_resource = boto3.resource('s3', region_name='us-east-1')

model = TFCLIPModel.from_pretrained(LAMBDA_TASK_ROOT)
tokenizer = CLIPTokenizer.from_pretrained(LAMBDA_TASK_ROOT)


def lambda_handler(event, context):
    
    faiss_content = s3_resource.Object(BUCKET_METADATA_NAME, FAISS_INDEX).get()['Body'].read()
    index_to_id = json.loads(s3_resource.Object(BUCKET_METADATA_NAME, INDEX_TO_ID).get()['Body'].read().decode('utf-8'))
    
    faiss_index = faiss.deserialize_index(np.frombuffer(faiss_content, dtype=np.uint8))
    print(f'** FAISS index loaded with size {faiss_index.ntotal} and embedding dimensionality {faiss_index.d}')
    print(f'** {INDEX_TO_ID} mapping loaded with length {len(index_to_id)} elements')
    
    if event.get("queryStringParameters"):
        
        # Get the value of the "k" query parameter
        k = event["queryStringParameters"].get("k")
        
        if k is not None:
            k = int(k)
        else:
            k = 5

        # Get the value of the "nprobes" query parameter
        nprobes = event["queryStringParameters"].get("nprobes")
        if nprobes is not None:
            nprobes = int(nprobes)
        else:
            nprobes = 3

    else:
        # Handle the case where there are no query parameters
        print(f'** No parameters "k" and "nprobes" found, using default values')
        k = 5
        nprobes = 3
    
    user_input_message = event['body']
    inputs = tokenizer(user_input_message, padding=True, return_tensors="tf")
    text_features = model.get_text_features(**inputs).numpy().squeeze()
    text_features = text_features / np.linalg.norm(text_features)
    
    faiss_index.nprobe = nprobes  # hyperparam: set how many of nearest cells to search
    neigh_dist, neigh_ind = faiss_index.search(np.array([text_features]), k)

    index_to_id = {int(k):v for k,v in index_to_id.items()}
    image_ids = [index_to_id[idx] for idx in neigh_ind.squeeze()]
    s3_urls = [s3.generate_presigned_url('get_object', Params={'Bucket': BUCKET_IMAGE_NAME, 'Key': im}) for im in image_ids]
    
    resp = {
        "neighbors": dict(list(zip(neigh_ind.squeeze().tolist(), neigh_dist.squeeze().tolist()))),
        "image_urls": s3_urls
    }
    
    return {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
        },
        "body": json.dumps(resp)
    }
   
