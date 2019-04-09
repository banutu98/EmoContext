import keras
import keras.layers as layers
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.engine import Layer
from keras.models import Model
import os

from utils import parse_file


class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='default',
                           )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.dimensions


def build_model():
    input_text = layers.Input(shape=(1,), dtype="string")
    embedding = ElmoEmbeddingLayer()(input_text)
    dense = layers.Dense(256, activation='relu')(embedding)
    pred = layers.Dense(4, activation='softmax')(dense)

    model = Model(inputs=[input_text], outputs=pred)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


# Reduce logging output.
if __name__ == '__main__':
    # Initialize session
    sess = tf.Session()
    K.set_session(sess)

    tf.logging.set_verbosity(tf.logging.ERROR)

    train_text = parse_file.get_sentences('../input_data/train.txt')
    train_text = np.array([[' <eos> '.join(tr)] for tr in train_text], dtype=object)
    # train_text, test_text = train_text[:100 * 32], train_text[-100 * 32:]
    # print(len(train_text))
    test_text = parse_file.get_sentences('../input_data/test.txt')
    test_text = np.array([[' <eos> '.join(tr)] for tr in test_text], dtype=object)
    train_label = parse_file.get_labels('../input_data/train.txt')
    test_label = parse_file.get_labels('../input_data/test.txt')
    normalized_train_label = list()
    normalized_test_label = list()
    for label in train_label:
        if label == 'happy':
            normalized_train_label.append(0)
        elif label == 'sad':
            normalized_train_label.append(1)
        elif label == 'angry':
            normalized_train_label.append(2)
        else:
            normalized_train_label.append(3)
    for label in test_label:
        if label == 'happy':
            normalized_test_label.append(0)
        elif label == 'sad':
            normalized_test_label.append(1)
        elif label == 'angry':
            normalized_test_label.append(2)
        else:
            normalized_test_label.append(3)
    del train_label
    train_label, test_label = normalized_train_label, normalized_test_label
    # list(normalized_train_label[:100 * 32]), list(normalized_train_label[-100 * 32:])
    train_label, test_label = keras.utils.to_categorical(train_label, num_classes=4), \
                              keras.utils.to_categorical(test_label, num_classes=4)
    model = build_model()
    if not os.path.exists('ElmoModel.h5'):
        model.fit(train_text,
                  train_label,
                  epochs=5,
                  batch_size=64)
        model.save_weights('ElmoModel.h5')
    else:
        model.load_weights('ElmoModel.h5')
    loss, accuracy = model.evaluate(test_text, test_label)
    print(loss, accuracy)
    conv = np.array([['You are annoying <eos> i am ILL let me have this <eos> Bhakkk bhosdi'],  # angry
                     ["Fight <eos> I don't fight. <eos> What are you doing"],  # others
                     ["Skydiving <eos> Especially while skydiving.. <eos> Miss u lot"],  # sad
                     ['stress where is that come from <eos> few media outlets in America. Next 24 hours <eos> i am enjoying'],  # happy
                     ["Please <eos> Okay........... BUT U DON'T NEED TO I LOVE DRAWING YOU THINGS <eos> I am also love drawing"],  # others
                     ["U r my lifee <eos> I'm not eww you're confusing me with you <eos> U r stupid"],  # angry
                     ["Not in mood <eos> when are you <eos> Leave yaar"],  # sad
                     ["Give me a reason <eos> because you're sick and you're not getting better <eos> Haha so funny"]  # happy
                     ], dtype=object)
    for c in conv:
        result = model.predict(c)
        idx = np.argmax(result)
        print(idx)
        if idx == 0:
            print('happy')
        elif idx == 1:
            print('sad')
        elif idx == 2:
            print('angry')
        else:
            print('others')
    # pre_save_preds = model.predict(test_text[:100])
    # print(pre_save_preds)
