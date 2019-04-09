import pickle

import keras
import keras.layers as layers
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import LSTM, Embedding, Dense, LeakyReLU, Lambda
from keras.layers.merge import add
from keras.models import Model

import preprocesing
from embeddings.elmo_legit import ElmoEmbeddingLayer as ElmoEmbeddingLayer
from utils import parse_file, F1_score
import os


def get_glove_matrix():
    with open(r'D:\PythonPrograms\IILN_EmoContext\embeddings\glove.pickle', 'rb') as f:
        data = pickle.load(f)
        return [v for v in data.values()]


def get_vocab_list():
    with open(r'D:\PythonPrograms\IILN_EmoContext\embeddings\glove.pickle', 'rb') as f:
        data = pickle.load(f)
        return [k for k in data]


def build_model():
    input_text = layers.Input(shape=(1,), dtype="string")
    elmo_embedding = ElmoEmbeddingLayer()(input_text)
    elmo_dense = Dense(64, activation='relu')(elmo_embedding)

    glove_matrix = np.array(get_glove_matrix())
    vocab_list = tf.constant(get_vocab_list())
    vocab_length = len(glove_matrix)

    lookup_table_op = tf.contrib.lookup.index_table_from_tensor(
        mapping=vocab_list,
        num_oov_buckets=0,
        default_value=0,
    )
    lambda_output = Lambda(lookup_table_op.lookup, output_shape=(1,))(input_text)
    glove_embedding = Embedding(input_dim=vocab_length,
                                output_dim=100,
                                weights=[glove_matrix],
                                input_length=1)(lambda_output)
    glove_lstm = LSTM(1024, dropout=0.2, recurrent_dropout=0.2, activation='softsign')(glove_embedding)
    glove_dense = Dense(64, activation='relu')(glove_lstm)

    fusion = add([elmo_dense, glove_dense])
    fusion_next = LeakyReLU()(fusion)
    fusion_final = Dense(64, activation='relu')(fusion_next)
    pred = layers.Dense(4, activation='softmax')(fusion_final)

    model = Model(inputs=[input_text], outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy', F1_score.f1])
    model.summary()
    return model


def get_data():
    train_text = preprocesing.parse_file.get_sentences('input_data/train.txt')
    train_text = preprocesing.conversations_to_sentences(preprocesing.get_normalized_conversations(train_text))
    train_text = np.array(train_text, dtype=object)

    test_text = preprocesing.parse_file.get_sentences('input_data/test.txt')
    test_text = preprocesing.conversations_to_sentences(preprocesing.get_normalized_conversations(test_text))
    test_text = np.array(test_text, dtype=object)

    train_label = preprocesing.get_normalized_labels(parse_file.get_labels('input_data/train.txt'))
    train_label = keras.utils.to_categorical(train_label, num_classes=4)

    test_label = preprocesing.get_normalized_labels(parse_file.get_labels('input_data/test.txt'))
    test_label = keras.utils.to_categorical(test_label, num_classes=4)

    return train_text, test_text, train_label, test_label


def get_dev_data():
    dev_text = preprocesing.parse_file.get_sentences('input_data/dev.txt')
    dev_text = preprocesing.conversations_to_sentences(preprocesing.get_normalized_conversations(dev_text))
    dev_text = np.array(dev_text, dtype=object)

    dev_label = preprocesing.get_normalized_labels(parse_file.get_labels('input_data/dev.txt'))
    dev_label = keras.utils.to_categorical(dev_label, num_classes=4)

    return dev_text, dev_label


def main():
    sess = tf.Session()
    K.set_session(sess)

    train_text, test_text, train_label, test_label = get_data()
    model = build_model()

    with sess.as_default():
        tf.tables_initializer().run(session=sess)
        if not os.path.exists('models/final_model_adadelta_88.h5'):
            model.fit(train_text,
                      train_label,
                      epochs=5,
                      batch_size=64)
            model.save_weights('models/final_model_adadelta_88.h5')
        else:
            model.load_weights('models/final_model_adadelta_88.h5')
        loss, accuracy, f1 = model.evaluate(test_text, test_label)
        dev_text, dev_label = get_dev_data()
        dev_loss, dev_accuracy, dev_f1 = model.evaluate(dev_text, dev_label)
        print(loss, accuracy, f1)
        print(f"Loss on dev data : {dev_loss} \t Accuracy on dev data: {dev_accuracy} "
              f"\t F1 Score on dev data: {dev_f1}")


def evaluate_conversation(conversation, model_file):
    sess = tf.Session()
    K.set_session(sess)
    model = build_model()
    normalized_conversation = preprocesing.conversations_to_sentences(
        preprocesing.get_normalized_conversations([conversation]))
    normalized_conversation = np.array(normalized_conversation, dtype=object)
    with sess.as_default():
        tf.tables_initializer().run(session=sess)
        model.load_weights(model_file)
        result = model.predict(normalized_conversation)
        idx = np.argmax(result)
        if idx == 0:
            return 'happy'
        elif idx == 1:
            return 'sad'
        elif idx == 2:
            return 'angry'
        else:
            return 'others'


if __name__ == '__main__':
    main()
