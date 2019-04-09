from nltk.corpus import brown
from gensim.models import Word2Vec
import multiprocessing

EMB_DIM = 100


def main():
    sentences = brown.sents()
    print(sentences)
    w2v = Word2Vec(sentences, size=EMB_DIM, window=5, min_count=5, negative=15, iter=10,
                   workers=multiprocessing.cpu_count())
    word_vectors = w2v.wv
    #print(word_vectors.similar_by_word('Saturday')[:3])


if __name__ == '__main__':
    main()
