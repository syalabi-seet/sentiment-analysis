import os
import re
import ast
import tarfile
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd

from transformers import (
    TFAutoModelForQuestionAnswering, 
    TFAutoModelForSequenceClassification,
    AutoConfig)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def seed_everything(SEED):
   random.seed(SEED)
   np.random.seed(SEED)
   tf.random.set_seed(SEED)
   print("INFO -- Random seed set.")


def get_data(input_path, max_len):
    train_data = pd.read_csv(input_path)
    for col in train_data.columns[3:]:
        train_data[col] = train_data[col].apply(ast.literal_eval).apply(np.array)
        if col in train_data.columns[3:-2]:
            train_data[col] = train_data[col].apply(lambda x: x[:max_len])
    return train_data


def get_input_arrays(df, type, batch_size, buffer_size, training_set):
    ids = np.vstack(df['input_ids'])
    mask = np.vstack(df['attention_mask'])
    start = np.vstack(df['start_token'])
    end = np.vstack(df['end_token'])
    sentiment = np.vstack(df['sentiment_token'])
    ids_PO = np.vstack(df['input_ids2'])
    mask_PO = np.vstack(df['attention_mask2'])
    sentiment_PO = df['sentiment'].astype('int8')

    if type=="QA":
        gen = tf.data.Dataset.from_tensor_slices((
            {
                "input_1": ids,
                "input_2": mask,
                "input_3": sentiment
            },
            {
                "output_1": tf.convert_to_tensor(start, dtype=tf.int32),
                "output_2": tf.convert_to_tensor(end, dtype=tf.int32)
            }
        ))

    else:
        gen = tf.data.Dataset.from_tensor_slices((
            {
                "input_1": ids_PO,
                "input_2": mask_PO
            }, tf.one_hot(sentiment_PO, depth=3)
        ))

    gen = gen.shuffle(buffer_size=256)
    gen = gen.batch(batch_size=batch_size)
    gen = gen.prefetch(buffer_size=buffer_size)
    return gen


def QAModel(pretrained, max_len):
    input_ids = tf.keras.layers.Input(shape=(max_len,), name="input_1",  dtype=tf.int32)
    att_mask = tf.keras.layers.Input(shape=(max_len,), name="input_2", dtype=tf.int32)
    sent_mask = tf.keras.layers.Input(shape=(max_len,), name="input_3", dtype=tf.int32)
    
    config = AutoConfig.from_pretrained(
        pretrained, 
        output_attention=True, 
        output_hidden_states=True, 
        use_cache=True)

    enc = TFAutoModelForQuestionAnswering.from_pretrained(
        pretrained, config=config)
    x = enc(input_ids, attention_mask=att_mask, token_type_ids=sent_mask)

    x1 = tf.keras.layers.Dropout(0.1)(x[0])
    x1 = tf.expand_dims(x1, axis=-1)
    x1 = tf.keras.layers.Conv1D(1,1)(x1)
    x1 = tf.keras.layers.Flatten()(x[0])
    x1 = tf.keras.layers.Activation('softmax')(x1)

    x2 = tf.keras.layers.Dropout(0.1)(x[1])
    x2 = tf.expand_dims(x2, axis=-1)
    x2 = tf.keras.layers.Conv1D(1,1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.Model(inputs=[input_ids, att_mask, sent_mask], outputs=[x1,x2])

    for layer in model.layers[3:4]:
        layer.trainable = True

    return model


def POModel(max_len, pretrained):
    input_ids = tf.keras.layers.Input(shape=(max_len,), name='input_1', dtype=tf.int32)
    att_mask = tf.keras.layers.Input(shape=(max_len,), name='input_2', dtype=tf.int32)

    enc = TFAutoModelForSequenceClassification.from_pretrained(pretrained, num_labels=3)
    x = enc(input_ids, attention_mask=att_mask)[0]

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation=None)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs=[input_ids, att_mask], outputs=x)

    for layer in model.layers[:3]:
        layer.trainable = True

    return model


def QAtraining(epochs, max_len, learning_rate, train_gen, val_gen):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=learning_rate, 
            first_decay_steps=3,
            t_mul=2.0, m_mul=1.0, alpha=0.0))

    cce_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.2)
    model = QAModel(pretrained='bert-base-uncased', max_len=max_len)
    t_jaccard_metric = tf.keras.metrics.MeanIoU(num_classes=2)
    v_jaccard_metric = tf.keras.metrics.MeanIoU(num_classes=2)
    metrics = ['loss', 'jaccard', 'val_loss', 'val_jaccard']
    best_score = 0

    t_jaccard_metric.reset_states()
    v_jaccard_metric.reset_states()

    @tf.function
    def getArrays(array):
        start_array = tf.where(
            tf.sequence_mask(
                lengths=tf.math.argmax(array[0], axis=1), 
                maxlen=max_len), 
            x=1, y=0)
        end_array = tf.where(
            tf.sequence_mask(
                lengths=tf.math.argmax(array[1], axis=1),
                maxlen=max_len),
            x=1, y=0)
        return tf.abs(end_array - start_array)

    @tf.function
    def train_step(X, y, loss_fn):
        y = [y['output_1'], y['output_2']]
        with tf.GradientTape() as tape:        
            y_hat = model(X, training=True)
            loss_value = loss_fn(y, y_hat)
            loss_value += sum(model.losses)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        t_jaccard_metric.update_state(getArrays(y), getArrays(y_hat))
        return loss_value

    @tf.function
    def val_step(X, y, loss_fn):
        y = [y['output_1'], y['output_2']]
        y_hat = model(X, training=False)
        loss_value = loss_fn(y, y_hat)
        v_jaccard_metric.update_state(getArrays(y), getArrays(y_hat))
        return loss_value

    i = 0
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        progbar = tf.keras.utils.Progbar(
            len(train_gen), interval=0.5, stateful_metrics=metrics)
        print(f"INFO -- Learning rate set to {optimizer.lr(epoch)}.")
        for step, (X_train, y_train) in enumerate(train_gen):
            train_loss = train_step(X_train, y_train, cce_fn)
            progbar.update(step, values=[('loss', train_loss), ('jaccard', t_jaccard_metric.result())])

        for x_batch_val, y_batch_val in val_gen:
            val_loss = val_step(x_batch_val, y_batch_val, cce_fn)

        values = [
            ('loss', train_loss), 
            ('jaccard', t_jaccard_metric.result()),
            ('val_loss', val_loss),
            ('val_jaccard', v_jaccard_metric.result())]
        progbar.update(len(train_gen), values=values, finalize=True)

        if best_score < v_jaccard_metric.result():
            best_score = v_jaccard_metric.result()
            best_model = model
            print("INFO -- Model improved.\n")
            i = 0
        else:
            i += 1
            if (i < 2) & (v_jaccard_metric.result() < t_jaccard_metric.result()):
                print("INFO -- Model did not improve.\n")
            else:
                print("INFO -- Early Stopping.\n")
                break

    return best_model, best_score


def POtraining(epochs, max_len, learning_rate, train_gen, val_gen):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate)

    cce_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.2)
    model = POModel(pretrained='distilroberta-base', max_len=max_len)
    t_f1_metric = tfa.metrics.F1Score(num_classes=3, average='micro')
    v_f1_metric = tfa.metrics.F1Score(num_classes=3, average='micro')
    metrics = ['loss', 'f1', 'val_loss', 'val_f1']
    best_score = 0

    t_f1_metric.reset_states()
    v_f1_metric.reset_states()


    @tf.function
    def train_step(X, y, loss_fn):
        with tf.GradientTape() as tape:
            y_hat = model(X, training=True)
            loss_value = loss_fn(y, y_hat)
            loss_value += sum(model.losses)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        t_f1_metric.update_state(y, y_hat)
        return loss_value

    @tf.function
    def val_step(X, y, loss_fn):
        y_hat = model(X, training=False)
        loss_value = loss_fn(y, y_hat)
        v_f1_metric.update_state(y, y_hat)
        return loss_value

    i = 0
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        progbar = tf.keras.utils.Progbar(
            len(train_gen), interval=0.5, stateful_metrics=metrics)
        for step, (X_train, y_train) in enumerate(train_gen):
            train_loss = train_step(X_train, y_train, cce_fn)
            progbar.update(step, values=[('loss', train_loss), ('f1', t_f1_metric.result())])

        for x_batch_val, y_batch_val in val_gen:
            val_loss = val_step(x_batch_val, y_batch_val, cce_fn)

        values = [
            ('loss', train_loss),
            ('f1', t_f1_metric.result()),
            ('val_loss', val_loss),
            ('val_f1', v_f1_metric.result())]
        progbar.update(len(train_gen), values=values, finalize=True)

        if ((t_f1_metric.result() - v_f1_metric.result()) < 0.02) & (v_f1_metric.result() > best_score):
            best_score = v_f1_metric.result()
            best_model = model
            print("INFO -- Model improved.\n")
            i = 0
        else:
            i += 1
            if i < 2:
                print("INFO -- Model did not improve.\n")
            else:
                print("INFO -- Early Stopping.\n")
                break

    return best_model, best_score
    

def save_tarfile(outfile, model_dir):
    with tarfile.open(outfile, "w:gz") as tar:
        for file in os.listdir(model_dir):
            tar.add(
                name=os.path.join(model_dir, file), 
                arcname=file,
                recursive=False)
        tar.close()