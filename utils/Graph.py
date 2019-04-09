import pickle
from sklearn.decomposition import PCA
from matplotlib import pyplot


def main():
    word_vectors = pickle.load(open("glove.pickle", "rb"))
    pca = PCA(n_components=2)
    # print(word_vectors.values())
    result = pca.fit_transform(list(word_vectors.values())[:50])
    pyplot.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(list(word_vectors.keys())[:50]):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()


if __name__ == '__main__':
    main()
