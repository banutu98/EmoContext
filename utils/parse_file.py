import nltk
import string


def get_words(file):
    result = list()
    happy, sad, angry, others = list(), list(), list(), list()
    with open(file, 'rb') as f:
        for line in f.readlines()[1:]:
            line_elements = line.strip(b'\n').replace(b'\t', b' ').replace(b'\r', b'').replace(b'  ', b' ').split(b' ')
            for element in line_elements[1:-1]:
                word = str(element, encoding='utf-8')
                if len(word) > 0:
                    result.append(word)
                    s = str(line_elements[-1], encoding='utf-8') + ".append(word)"
                    eval(s)
    return happy, sad, angry, others, result


def get_words_from_conversation(conversation):
    result = list()
    for sentence in conversation:
        sentence = sentence.strip()
        result.extend(sentence.split(' '))
    return result


def get_common_nouns(sentence):
    text = nltk.word_tokenize(sentence)
    pos = nltk.pos_tag(text)
    result = list()
    for i in pos:
        if i[1] == 'NN' and i[0][0].islower():
            result.append(i[0])
    return result


def get_labels(file):
    result = list()
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            line_elements = line.strip('\n').replace('\r', '').replace('  ', ' ').split('\t')
            label = line_elements[-1]
            result.append(label)
    return result


def get_sentences_by_label(file):
    happy = list()
    sad = list()
    angry = list()
    others = list()
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            line_elements = line.strip('\n').replace('\r', '').replace('  ', ' ').split('\t')
            label = line_elements[-1]
            line_elements = [sentence for sentence in line_elements[1:-1]]
            eval("{}.append({})".format(label, line_elements))
    return [happy, sad, angry, others]


def get_sentences(file):
    result = list()
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            line_elements = line.strip('\n').replace('\r', '').replace('  ', ' ').split('\t')
            line_elements = [sentence for sentence in line_elements[1:-1]]
            result.append(line_elements)
    return result


def get_conversations(file):
    result = list()
    with open(file, 'rb') as f:
        for line in f.readlines()[1:]:
            line_elements = line.strip(b'\n').replace(b'\r', b'').replace(b'  ', b' ').split(b'\t')
            line_elements = [str(sentence, encoding='utf-8') for sentence in line_elements[1:-1]]
            for i in range(len(line_elements)):
                line_elements[i] = line_elements[i].translate(str.maketrans('', '', string.punctuation))
            result.append(line_elements)
    return result


if __name__ == '__main__':
    pass
