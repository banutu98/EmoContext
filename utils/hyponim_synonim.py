import nltk
from nltk.corpus import wordnet


def get_related_words(word):
    synonyms, antonyms = list(), list()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return synonyms, antonyms


def get_hyponim(word):
    pos = nltk.pos_tag(word)
    print(pos)
    if pos[0][1].startswith('J'):
        pos = wordnet.ADJ
    elif pos[0][1].startswith('V'):
        pos = wordnet.VERB
    elif pos[0][1].startswith('N'):
        pos = wordnet.NOUN
    elif pos[0][1].startswith('R'):
        pos = wordnet.ADV
    else:
        return []
    # print(index)
    final_word = word[0].lower() + '.' + pos + '.01'
    try:
        syn = wordnet.synset(final_word)
    except Exception:
        return []
        # print(final_word)
    return syn.hypernyms()


if __name__ == '__main__':
    get_hyponim(['dog'])
