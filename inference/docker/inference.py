import os
import ast
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
        tar.extractall(path="/opt/ml/processing/model/")

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
                "SELECT user_id, text " \
                "FROM sentiment_analysis " \
                "WHERE sentiments IS NOT NULL;")
            results = cursor.fetchall()
        print("INFO -- Input data extracted from sentiment_analysis.")  

        # Main loop
        outputs = {'user_id': [], 'texts': [], 'sentiments': [], 'selected_texts': []}
        all_sentiments = []
        all_selected_texts = []
        for user_id, texts in results:
            texts = ast.literal_eval(texts)
            sentiments, selected_texts = get_outputs(
                texts=texts, 
                polarity_tokenizer=tokenizer, 
                phrase_tokenizer=tokenizer,
                polarity_model=polarity_model, 
                phrase_model=phrase_model,
                POmax_len=args.POmax_len,
                QAmax_len=args.QAmax_len)
            outputs['user_id'].append(user_id)
            outputs['texts'].append(texts)
            outputs['sentiments'].append(sentiments)
            outputs['selected_texts'].append(selected_texts)

            # Output
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE sentiment_analysis " \
                    "SET sentiments = %s, selected_texts = %s " \
                    "WHERE user_id = %s;",
                    (str(sentiments), str(selected_texts), user_id))
            
            connection.commit()
            print("INFO -- Output updated in sentiment_analysis.")

        with open('/opt/ml/processing/output/output.json', 'w') as f:
            json.dump(outputs, f, indent=4)        

    print("INFO -- Job Completed.")