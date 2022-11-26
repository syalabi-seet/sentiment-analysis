import os
import time
import json
import argparse
import pandas as pd
import tarfile
import sys
import subprocess
import psycopg2

import tensorflow as tf
from transformers import AutoTokenizer
import transformers

from helper_functions import *

# Mute warnings
tf.get_logger().setLevel('ERROR')

print(subprocess.check_output('nvcc --version'.split(' ')).decode())
print(sys.version)
print(tf.__version__)
print(transformers.__version__)
print(pd.__version__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="For S3 bucket access")
    parser.add_argument("--access_id", dest='access_id', type=str)
    parser.add_argument("--access_key", dest='access_key', type=str)
    parser.add_argument("--num_classes", dest='num_classes', type=int, default=3)
    parser.add_argument("--QAmax_len", dest='QAmax_len', type=int, default=64)
    parser.add_argument("--POmax_len", dest='POmax_len', type=int, default=128) 
    parser.add_argument("--bucket_name", dest='bucket_name', type=str, default='syalabi-bucket')

    parser.add_argument("--host", dest="host", type=str)
    parser.add_argument("--port", dest="port", type=str)
    parser.add_argument("--user", dest="user", type=str)
    parser.add_argument("--password", dest="password", type=str)
    parser.add_argument("--dbname", dest="dbname", type=str)

    args, _ = parser.parse_known_args()

    tokenizer = AutoTokenizer.from_pretrained(
        'roberta-base',
        add_prefix_space=True)
    
    print("INFO -- Tokenizers initialized.")

    model_path = os.path.join(
        "/opt/ml/processing/model/sentiment_models.tar.gz")

    with tarfile.open(model_path) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path="/opt/ml/processing/model/")

    polarity_model = PolarityModel(
        model_path='/opt/ml/processing/model/polarity_model.h5',
        max_len=args.POmax_len,
        num_classes=args.num_classes)

    print("INFO -- Polarity model initialized.")
   
    phrase_model = PhraseModel(
        model_path='/opt/ml/processing/model/phrase_model.h5',
        max_len=args.QAmax_len)

    print("INFO -- Phrase model initialized.")    

    connection = psycopg2.connect(
        host=args.host,
        user=args.user,
        password=args.password,
        dbname=args.dbname,
        connect_timeout=5)

    # Input
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT feedback_id, COALESCE(reason_text, ''), COALESCE(suggestion_text, '') " \
                "FROM feedback " \
                "WHERE status = false;")
            results = cursor.fetchall()
        print("INFO -- Input data extracted from feedback table.")  

        # Main loop
        outputs = {'feedback_id': [], 'text': [], 'sentiment': [], 'selected_text': [], 'timestamp': []}
        for feedback_id, text, suggestion_text in results:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            text = " ".join([text.strip(), suggestion_text.strip()])
            sentiment, selected_text = get_outputs(
                text=text, 
                polarity_tokenizer=tokenizer, 
                phrase_tokenizer=tokenizer,
                polarity_model=polarity_model, 
                phrase_model=phrase_model,
                POmax_len=args.POmax_len,
                QAmax_len=args.QAmax_len)

            outputs['feedback_id'].append(feedback_id)
            outputs['text'].append(text)
            outputs['sentiment'].append(sentiment)
            outputs['selected_text'].append(selected_text)
            outputs['timestamp'].append(timestamp)            
           
            # Update status
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE feedback " \
                    "SET status = true " \
                    "WHERE feedback_id = %s;",
                    (feedback_id,)
                )

            # Insert outputs into feedback_analysis_response table
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO feedback_analysis_response(feedback_id, sentiment, selected_text, timestamp) " \
                    "VALUES (%s, %s, %s, %s);",
                    (str(feedback_id), sentiment, selected_text, timestamp))
            
        connection.commit()
        print("INFO -- Output updated in feedback_analysis_response table.")
        with open('/opt/ml/processing/output/output.json', 'w') as f:
            json.dump(outputs, f, indent=4)

    print("INFO -- Job Completed.")