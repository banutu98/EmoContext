from sklearn.feature_extraction.text import TfidfVectorizer


def get_conversations(file):
    result = list()
    with open(file, 'rb') as f:
        for line in f.readlines()[1:]:
            line_elements = line.strip(b'\n').replace(b'\r', b'').replace(b'  ', b' ').split(b'\t')
            line_elements = [str(sentence, encoding='utf-8') for sentence in line_elements[1:-1]]
            result.append(line_elements)
    return result


if __name__ == '__main__':
    conversations = get_conversations('train.txt')
    vectorizer = TfidfVectorizer()
    whole = list()
    for conversation in conversations:
        current_dict = dict()
        X = vectorizer.fit_transform(conversation)
        unique_words = vectorizer.get_feature_names()
        tf_idf_scores = X.toarray()
        for sentence in range(len(conversation)):
            print(conversation[sentence])
            splitted = conversation[sentence].split(' ')
            for word in range(len(splitted)):
                current_dict[splitted[word]] = tf_idf_scores[sentence][word]
        print(current_dict)
        break

    # print(sentences)
