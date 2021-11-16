import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import (
    BatchNormalization, Dense, Dropout,
    Input, Conv1D, Flatten, Activation)
from transformers import (AutoConfig, TFAutoModelForSequenceClassification,
    TFAutoModelForQuestionAnswering)


def PolarityTokenizer(text, tokenizer, max_length):
    """
        Tokenizes text using the polarity tokenizer and
        returns the input arrays for polarity model.

        :param text: list of str
        :param tokenizer: transformers.PreTrainedTokenizer object
        :return: list of arrays
    """
    inputs = tokenizer.encode_plus(
        text=text,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        is_split_into_words=False,
        return_tensors='tf',
        return_attention_mask=True)

    input_id = np.array(inputs['input_ids'].numpy()[0]).reshape((1, -1))
    att_mask = np.array(inputs['attention_mask'].numpy()[0]).reshape((1, -1))

    return [input_id, att_mask]


def PhraseTokenizer(text, sentiment, tokenizer, max_length):
    """
        Tokenizes text and sentiments and returns the
        input arrays for phrase model.

        :param text: list of str
        :param sentiment: list of str
        :param tokenizer: transformers.PreTrainedTokenizer object
        :return: list of arrays
    """
    inputs = tokenizer.encode_plus(
        text=text,
        text_pair=sentiment,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        is_split_into_words=False,
        return_tensors='tf',
        return_attention_mask=True)

    input_id = list(inputs['input_ids'].numpy()[0])
    att_mask = list(inputs['attention_mask'].numpy()[0])

    sent_enc = tokenizer.encode(
        text=sentiment,
        add_special_tokens=False)

    sent_idx = input_id.index(sent_enc[0])
    sent_mask = np.zeros(max_length, dtype='int32')
    sent_mask[sent_idx] = 1

    input_id = np.array(input_id).reshape((1, -1))
    att_mask = np.array(att_mask).reshape((1, -1))
    sent_mask = np.array(sent_mask).reshape((1, -1))

    return [input_id, att_mask, sent_mask]

def PhraseDecoder(input_ids, prediction, tokenizer):
    """
        Decodes predicted start and end tokens from phrase
        model and returns the raw text string form

        :param input_ids: array
        :param prediction: list of arrays
        :param tokenizer: transformers.PreTrainedTokenizer object
        :return: str
    """
    start_idx = np.argmax(prediction[0], axis=-1)[0]
    end_idx = np.argmax(prediction[1], axis=-1)[0]
    selected_text = input_ids[start_idx:end_idx]
    selected_text = tokenizer.decode(selected_text).strip()
    return selected_text


def get_outputs(text, polarity_tokenizer, phrase_tokenizer, polarity_model, phrase_model, POmax_len, QAmax_len):
    polarity_inputs = PolarityTokenizer(
        text=text,
        tokenizer=polarity_tokenizer,
        max_length=POmax_len)
    polarity_preds = polarity_model.predict(polarity_inputs)
    polarity_preds = np.argmax(polarity_preds, axis=1)
    if polarity_preds == 0:
        sentiment = 'negative'
    elif polarity_preds == 1:
        sentiment = 'neutral'
    else:
        sentiment = 'positive'            
    phrase_inputs = PhraseTokenizer(
        text=text,
        sentiment=sentiment,
        tokenizer=phrase_tokenizer,
        max_length=QAmax_len)
    phrase_ids = phrase_inputs[0][0]
    phrase_preds = phrase_model.predict(phrase_inputs)
    selected_text = PhraseDecoder(
        input_ids=phrase_ids, 
        prediction=phrase_preds, 
        tokenizer=phrase_tokenizer)
    return sentiment, selected_text


def PolarityModel(model_path, max_len, num_classes):
    """
    Returns the polarity model loaded with pretrained weights.

    :return: tf.keras.Model object
    """
    input_ids = tf.keras.layers.Input(shape=(max_len,), name='input_1', dtype=tf.int32)
    att_mask = tf.keras.layers.Input(shape=(max_len,), name='input_2', dtype=tf.int32)

    enc = TFAutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=3)
    x = enc(input_ids, attention_mask=att_mask)[0]

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation=None)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs=[input_ids, att_mask], outputs=x)

    for layer in model.layers[:3]:
        layer.trainable = True

    model.load_weights(model_path)

    return model


def PhraseModel(model_path, max_len):
    """
    Returns the phrase model loaded with pretrained weights.

    :return: tf.keras.Model object
    """
    input_ids = tf.keras.layers.Input(shape=(max_len,), name="input_1",  dtype=tf.int32)
    att_mask = tf.keras.layers.Input(shape=(max_len,), name="input_2", dtype=tf.int32)
    sent_mask = tf.keras.layers.Input(shape=(max_len,), name="input_3", dtype=tf.int32)
    
    config = AutoConfig.from_pretrained(
        'bert-base-uncased', 
        output_attention=True, 
        output_hidden_states=True, 
        use_cache=True)

    enc = TFAutoModelForQuestionAnswering.from_pretrained(
        'bert-base-uncased', config=config)
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

    model.load_weights(model_path)

    return model