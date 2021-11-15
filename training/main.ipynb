import sys
import argparse
import subprocess
import tensorflow as tf

from helper_functions import *

# Environ set-up
tf.get_logger().setLevel('ERROR')

print("------ Dependencies ------")
print(subprocess.check_output('nvcc --version'.split(' ')).decode())
print("Python version:", sys.version)
print("Tensorflow version:", tf.__version__)

# Mixed precision
try:
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("INFO -- Mixed precision set.")
except:
    print("ERROR -- Mixed precision not set.")

SEED = 42
seed_everything(SEED)
buffer_size = tf.data.AUTOTUNE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--access_id", dest="access_id", type=str)
    parser.add_argument(
        "--access_key", dest="access_key", type=str)
    parser.add_argument(
        "--max_length", dest="max_len", type=int, default=64)    
    parser.add_argument(
        "--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument(
        "--epochs", dest="epochs", type=int, default=20)
    parser.add_argument(
        "--lr", dest='lr', type=float, default=1e-5)
    parser.add_argument(
        "--splits", dest='splits', type=int, default=5)
    parser.add_argument(
        "--pretrained_QA", dest='pretrained_QA', type=str, 
        default='bert-base-uncased')
    parser.add_argument(
        "--pretrained_PO", dest='pretrained_PO', type=str, 
        default='distilroberta-base')   
    parser.add_argument(
        "--bucket_name", dest="bucket_name", type=str, 
        default="syalabi-bucket")

    args, _ = parser.parse_known_args()

    try: 
        train_data = get_data(
            input_path='/opt/ml/processing/input/train_cleaned.csv',
            max_len=args.max_len)
        print("INFO -- Successfully fetched data from S3 bucket.")
    except:
        print('ERROR -- Fetching data from S3 bucket has failed.')

    QA_best_score, PO_best_score = 0, 0
    skf = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=SEED)
    for fold, (train_idx, val_idx) in enumerate(skf.split(
            X=train_data.drop('sentiment', axis=1),
            y=train_data['sentiment'])):
        
        QAtrain = get_input_arrays(
            train_data.loc[train_idx, :].reset_index(drop=True), 
            type="QA",
            batch_size=args.batch_size,
            buffer_size=buffer_size,
            training_set=True)

        QAval = get_input_arrays(
            train_data.loc[val_idx, :].reset_index(drop=True),
            type="QA",
            batch_size=args.batch_size,
            buffer_size=buffer_size,
            training_set=False)

        POtrain = get_input_arrays(
            train_data.loc[train_idx, :].reset_index(drop=True), 
            type="PO",
            batch_size=args.batch_size,
            buffer_size=buffer_size,
            training_set=True)

        POval = get_input_arrays(
            train_data.loc[val_idx, :].reset_index(drop=True), 
            type="PO",
            batch_size=args.batch_size,
            buffer_size=buffer_size,
            training_set=False)

        # Phrase Loop
        print(f'INFO -- Phrase Training Fold {fold+1} of {args.splits}.')
        QAmodel, QA_val_score = QAtraining(
            epochs=args.epochs, 
            max_len=args.max_len,
            learning_rate=args.lr,
            train_gen=QAtrain,
            val_gen=QAval)
       
        if QA_val_score > QA_best_score:
            QA_best_score = QA_val_score
            QAmodel.save_weights('/opt/ml/processing/model/phrase_model.h5')
            print(f"INFO -- FOLD {fold+1} phrase model weights saved.")
        else:
            print("INFO -- Model weights were not replaced.")

        # Polarity Loop
        tf.keras.backend.clear_session()
        print(f'INFO -- Polarity Training Fold {fold+1} of {args.splits}.')
        
        POmodel, PO_val_score = POtraining(
            epochs=args.epochs, 
            max_len=args.max_len,
            learning_rate=args.lr,
            train_gen=POtrain,
            val_gen=POval)

        if PO_val_score > PO_best_score:
            PO_best_score = PO_val_score
            POmodel.save_weights('/opt/ml/processing/model/polarity_model.h5')
            print(f"INFO -- FOLD {fold+1} polarity model weights saved.")
        else:
            print("INFO -- Model weights were not replaced.")

    save_tarfile(
        outfile="/opt/ml/processing/output/sentiment_models.tar.gz",
        model_dir="/opt/ml/processing/model")
    print("INFO -- Successfully compressed models into tar.")
    print("Training completed.")