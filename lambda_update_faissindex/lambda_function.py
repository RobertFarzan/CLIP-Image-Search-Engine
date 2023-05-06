import os
import json
import boto3
import botocore
import pickle
import faiss
import numpy as np
from datetime import datetime

LAMBDA_TASK_ROOT = os.environ.get('LAMBDA_TASK_ROOT', '/var/task')
TABLE_NAME = "CLIPEmbeddingsTable"
BUCKET_NAME = "clip.index.metadata.tfm.robert"

# files
FAISS_INDEX = "faiss.index"
METADATA_FILE = "metadata.json"
INDEX_TO_ID = "index_to_id.json"

s3 = boto3.client('s3', region_name='us-east-1')
s3_resource = boto3.resource('s3', region_name='us-east-1')

dynamodb_resource = boto3.resource('dynamodb', region_name='us-east-1')
dynamodb_table = dynamodb_resource.Table(TABLE_NAME)

train_embs = np.load(os.path.join(LAMBDA_TASK_ROOT, 'image_CLIP_train_embeddings.npy'))

def lambda_handler(event, context):
    
    # check if metadata exists in the bucket
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=METADATA_FILE)
        print(f'** {METADATA_FILE} already exists in {BUCKET_NAME}.')
        
        metadata_content = json.loads(s3_resource.Object(BUCKET_NAME, METADATA_FILE).get()['Body'].read().decode('utf-8'))
        print(f'** {METADATA_FILE} file loaded with content\t{metadata_content}')
        
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f'{METADATA_FILE} does not exist in {BUCKET_NAME}, creating...')
            
            metadata_content = {'last_timestamp': datetime.utcfromtimestamp(0).isoformat(timespec='microseconds'),
                                'last_index': 0
                                }
            
            # Create the file in the S3 bucket
            s3_resource.Object(BUCKET_NAME, METADATA_FILE).put(Body=json.dumps(metadata_content))
            print(f'{METADATA_FILE} created in {BUCKET_NAME}')
        else:
            print('Error:', e)
    
    
    # check if FAISS index exists in the bucket
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=FAISS_INDEX)
        print(f'** {FAISS_INDEX} already exists in {BUCKET_NAME}.')
        
        faiss_content = s3_resource.Object(BUCKET_NAME, FAISS_INDEX).get()['Body'].read()
        faiss_index = faiss.deserialize_index(np.frombuffer(faiss_content, dtype=np.uint8))
        
        print(f'** FAISS index loaded with size {faiss_index.ntotal} and embedding dimensionality {faiss_index.d}')
        
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f'{FAISS_INDEX} does not exist in {BUCKET_NAME}, creating...')
            
            nlist = 15
            d = train_embs.shape[1] # == 512
            k = 10

            quantizer = faiss.IndexFlatIP(d)  # how the vectors will be stored/compared
            faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist)
            faiss_index.train(train_embs)  # we must train the index to cluster into cells
            
            s3_resource.Object(BUCKET_NAME, FAISS_INDEX).put(Body=faiss.serialize_index(faiss_index).tobytes())
            print(f'{FAISS_INDEX} created in {BUCKET_NAME}')
        else:
            print('Error:', e)
            
            
    # check if index_to_id.json index exists in the bucket
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=INDEX_TO_ID)
        print(f'** {INDEX_TO_ID} already exists in {BUCKET_NAME}.')
        
        index_to_id = json.loads(s3_resource.Object(BUCKET_NAME, INDEX_TO_ID).get()['Body'].read().decode('utf-8'))
        
        print(f'** {INDEX_TO_ID} mapping loaded with length {len(index_to_id)} elements')
        
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f'{INDEX_TO_ID} does not exist in {BUCKET_NAME}, creating...')
            
            index_to_id = {}
            
            s3_resource.Object(BUCKET_NAME, INDEX_TO_ID).put(Body=json.dumps(index_to_id))
            print(f'{INDEX_TO_ID} created in {BUCKET_NAME}')
        else:
            print('Error:', e)
    
    # call to DynamoDB to update embeddings
    timestamp_cutoff = metadata_content['last_timestamp']
    partition_key = 'image'
    
    query_params = {
                        'KeyConditionExpression': '#pk = :pk AND #ts > :ts',
                        'ExpressionAttributeNames': {
                             '#pk': 'type_pk',
                             '#ts': 'timestamp'
                        },
                        'ExpressionAttributeValues': {
                            ':pk': partition_key,
                            ':ts': timestamp_cutoff
                        },
                        'ScanIndexForward': False
                    }

    results = []
    last_evaluated_key = None

    while True:
        if last_evaluated_key:
            query_params['ExclusiveStartKey'] = last_evaluated_key

        response = dynamodb_table.query(**query_params)
        items = response['Items']
        results += items

        last_evaluated_key = response.get('LastEvaluatedKey')
        if not last_evaluated_key:
            break


    print(f"** [DynamoDB] Retrieved {len(results)} new items from query with base timestamp {timestamp_cutoff}")
    
    # use query results to update FAISS index
    results = sorted(results, key=lambda x: x['timestamp'], reverse=True)
    
    if results:
        # update metadata
        metadata_content['last_timestamp'] = results[0]['timestamp']
        metadata_content['last_index'] = metadata_content['last_index'] + len(results)

        id_to_index = {v:k for k,v in index_to_id.items()}

        for item in results:
            image_id = item['image_id']
            if image_id not in id_to_index:
                embedding = pickle.loads(item['embedding'].value)
                faiss_index.add_with_ids(embedding.reshape(1, -1), np.array([faiss_index.ntotal]))
                index_to_id[int(faiss_index.ntotal - 1)] = image_id
            else:
                print(f"** Image with ID {image_id} already present in FAISS index")

        # write back the updated results to S3
        s3_resource.Object(BUCKET_NAME, METADATA_FILE).put(Body=json.dumps(metadata_content))
        s3_resource.Object(BUCKET_NAME, FAISS_INDEX).put(Body=faiss.serialize_index(faiss_index).tobytes())
        s3_resource.Object(BUCKET_NAME, INDEX_TO_ID).put(Body=json.dumps(index_to_id))
        print(f'** Files {METADATA_FILE}, {FAISS_INDEX}, {INDEX_TO_ID} updated and uploaded to S3')
    
        return {
            'statusCode': 200,
            'body': json.dumps(f'FAISS index updated successfully with {len(results)} new elements.')
        }
    
    else:
        return {
            'statusCode': 200,
            'body': json.dumps(f'No new records found to update FAISS index.')
        }