import pickle

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np

from utils import parse_file


def remove_stopwords(conversations):
    stop_words = set(stopwords.words('english'))
    for conversation in conversations:
        for sentence in conversation:
            for word in sentence.split(" "):
                if word not in stop_words:
                    sentence = sentence.replace(word, '')
    return conversations


def lemmatize(conversations):
    wordnet_lemmatizer = WordNetLemmatizer()
    for conversation in conversations:
        for sentence in conversation:
            for word in sentence.split(" "):
                lemma = wordnet_lemmatizer.lemmatize(word)
                sentence = sentence.replace(word, lemma)
    return conversations


def main():
    conversations = lemmatize(remove_stopwords(parse_file.get_conversations("../input_data/train.txt")))
    glove_input_file = '../resources/glove.6B.100d.txt'
    word2vec_output_file = 'glove.6B.100d.txt.word2vec'
    glove2word2vec(glove_input_file, word2vec_output_file)
    # load the Stanford GloVe model
    filename = 'glove.6B.100d.txt.word2vec'
    model = KeyedVectors.load_word2vec_format(filename, binary=False)
    word_vectors = dict()
    for conversation in conversations:
        for sentence in conversation:
            for word in sentence.split(" "):
                try:
                    word_vec = model.get_vector(word.lower())
                    word_vectors[word.lower()] = word_vec
                except KeyError:
                    word_vectors[word.lower()] = np.random.randn(100)
    print(word_vectors)
    with open('glove.pickle', 'wb') as f:
        pickle.dump(word_vectors, f)


if __name__ == '__main__':
    main()
