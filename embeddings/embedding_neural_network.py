from nltk.corpus import conll2000
from gensim.models import Word2Vec
import tensorflow as tf
from keras.layers import Dense, Embedding, Activation, Flatten
from keras import Sequential
from keras.utils import to_categorical
import numpy as np
import collections
import multiprocessing

EMB_DIM = 300
UNK_INDEX = 0
UNK_TOKEN = 'UNK'
EOS_INDEX = 1
EOS_TOKEN = 'EOS'

HIDDEN_SIZE = 50
BATCH_SIZE = 128
CONTEXT_SIZE = 2


def add_new_word(new_word, new_vector, new_index, embedding_matrix, word2id):
    embedding_matrix = np.insert(embedding_matrix, [new_index], [new_vector], axis=0)
    word2id = {word: (index+1) if index >= new_index else index for word, index in word2id.items()}
    word2id[new_word] = new_index
    return embedding_matrix, word2id


def get_window_int_data(tagged_words, word2id, tag2id):
    x, y = list(), list()
    unk_count = 0

    span = 2 * CONTEXT_SIZE + 1
    buffer = collections.deque(maxlen=span)
    padding = [(EOS_TOKEN, None)] * CONTEXT_SIZE
    buffer += padding + tagged_words[:CONTEXT_SIZE]

    for item in (tagged_words[CONTEXT_SIZE:] + padding):
        buffer.append(item)
        window_ids = np.array([word2id.get(word) if word in word2id else UNK_INDEX for (word, _) in buffer])
        x.append(window_ids)
        middle_word, middle_tag = buffer[CONTEXT_SIZE]
        y.append(tag2id.get(middle_tag))
        if middle_word not in word2id:
            unk_count += 1
    return np.array(x), np.array(y)


def get_tag_vocabulary(tagged_words):
    tag2id = {}
    for item in tagged_words:
        tag = item[1]
        tag2id.setdefault(tag, len(tag2id))
    return tag2id


def evaluate_model(model, id2word, x_test, y_test):
    _, acc = model.evaluate(x_test, y_test)
    y_pred = model.predict_classes(x_test)
    error_counter = collections.Counter()
    for i in range(len(x_test)):
        correct_tag_id = np.argmax(y_test[i])
        if y_pred[i] != correct_tag_id:
            if isinstance(x_test[i], np.ndarray):
                word = id2word[x_test[i][CONTEXT_SIZE]]
            else:
                word = id2word[x_test[i]]
            error_counter[word] += 1
    print('Most common errors:\n', error_counter.most_common(10))


def define_context_sensitive_model(embedding_matrix, class_count):
    vocab_length = len(embedding_matrix)
    total_span = CONTEXT_SIZE * 2 + 1

    model = Sequential()
    model.add(Embedding(input_dim=vocab_length,
                        output_dim=EMB_DIM,
                        weights=[embedding_matrix],
                        input_length=total_span))
    model.add(Flatten())
    model.add(Dense(HIDDEN_SIZE))
    model.add(Activation('tanh'))
    model.add(Dense(class_count))
    model.add(Activation('softmax'))

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    train_words = conll2000.tagged_words("train.txt")
    test_words = conll2000.tagged_words("test.txt")
    sentences = conll2000.sents()
    w2v = Word2Vec(sentences, size=EMB_DIM, window=5, min_count=5, negative=15, iter=10,
                   workers=multiprocessing.cpu_count())
    word_vectors = w2v.wv
    word2id = {k: v.index for k, v in word_vectors.vocab.items()}
    tag2id = get_tag_vocabulary(train_words)
    x_train, y_train = get_window_int_data(train_words, word2id, tag2id)
    x_test, y_test = get_window_int_data(test_words, word2id, tag2id)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    embedding_matrix = word_vectors.vectors
    unk_vector = embedding_matrix.mean(0)
    eos_vector = np.random.standard_normal(EMB_DIM)
    embedding_matrix, word2id = add_new_word(UNK_TOKEN, unk_vector, UNK_INDEX, embedding_matrix, word2id)
    embedding_matrix, word2id = add_new_word(EOS_TOKEN, eos_vector, EOS_INDEX, embedding_matrix, word2id)

    pos_model = define_context_sensitive_model(embedding_matrix, len(tag2id))
    pos_model.summary()
    pos_model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1)

    id2word = sorted(word2id, key=word2id.get)
    evaluate_model(pos_model, id2word, x_test, y_test)


if __name__ == '__main__':
    main()
