from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from utils.parse_file import get_sentences, get_conversations
from embeddings.glove_embedding import lemmatize
import json
import nltk


def get_useful_information(file):
    st = StanfordNERTagger('stanford-ner-2018-10-16/classifiers/english.conll.4class.distsim.crf.ser.gz',
                           'stanford-ner-2018-10-16/stanford-ner.jar',
                           encoding='utf-8')
    conversations_with_punctuation = get_sentences(file)
    conversations_without_punctuation = get_conversations(file)
    total_tokens = 0
    total_tokens_without_punctuation = 0
    for conversation in conversations_with_punctuation[200:300]:
        for sentence in conversation:
            total_tokens += len(word_tokenize(sentence))

    resulted_entities = dict()
    total_sentences = 0
    i = 0
    conversations_without_punctuation = lemmatize(conversations_without_punctuation)
    for conversation in conversations_without_punctuation[200:300]:
        print(i)
        i += 1
        for sentence in conversation:
            total_sentences += 1
            tokenized_text = word_tokenize(sentence)
            total_tokens_without_punctuation += len(tokenized_text)
            classified_text = st.tag(tokenized_text)
            for c in classified_text:
                if c[1] not in resulted_entities:
                    resulted_entities[c[1]] = set(c[0])
                else:
                    resulted_entities[c[1]].add(c[0])
    return total_sentences, total_tokens, total_tokens_without_punctuation, resulted_entities


def get_useful_information2(file):
    conversations_with_punctuation = get_sentences(file)
    conversations_without_punctuation = get_conversations(file)
    total_tokens = 0
    total_tokens_without_punctuation = 0
    resulted_entities = dict()
    i = 0
    for conversation in conversations_with_punctuation:
        print(i)
        i += 1
        for sentence in conversation:
            total_tokens += len(word_tokenize(sentence))
            for sent in nltk.sent_tokenize(sentence):
                for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                    if hasattr(chunk, 'label'):
                        if chunk.label() not in resulted_entities:
                            resulted_entities[chunk.label()] = set(c[0] for c in chunk)
                        else:
                            resulted_entities[chunk.label()] = resulted_entities[chunk.label()].union(set(c[0] for c in chunk))
    total_sentences = 0
    conversations_without_punctuation = lemmatize(conversations_without_punctuation)
    for conversation in conversations_without_punctuation:
        i += 1
        for sentence in conversation:
            total_sentences += 1
            tokenized_text = word_tokenize(sentence)
            total_tokens_without_punctuation += len(tokenized_text)
    return total_sentences, total_tokens, total_tokens_without_punctuation, resulted_entities


def main():
    total_sentences, total_tokens, total_tokens_without_punctuation, resulted_entities = \
        get_useful_information('../train.txt')
    counts = dict()
    counts['Sentences'] = total_sentences
    counts['Total Tokens (punctuation included)'] = total_tokens
    counts['Total Tokens (excluded punctuation)'] = total_tokens_without_punctuation
    counts['Total Entities'] = sum([len(v) for k, v in resulted_entities.items() if k != 'O'])
    for key in resulted_entities:
        if key != 'O':
            counts[key] = len(resulted_entities[key])
    with open('entities_result_part2.json', 'w') as g, open('entities_part2.json', 'w') as h:
        json.dump(counts, g)
        del resulted_entities['O']
        for key in resulted_entities:
            resulted_entities[key] = list(resulted_entities[key])
        json.dump(resulted_entities, h)


def main2():
    total_sentences, total_tokens, total_tokens_without_punctuation, resulted_entities = \
        get_useful_information2('../train.txt')
    counts = dict()
    counts['Sentences'] = total_sentences
    counts['Total Tokens (punctuation included)'] = total_tokens
    counts['Total Tokens (excluded punctuation)'] = total_tokens_without_punctuation
    counts['Total Entities'] = sum([len(v) for k, v in resulted_entities.items() if k != 'O'])
    for key in resulted_entities:
        if key != 'O':
            counts[key] = len(resulted_entities[key])
    with open('statistics_result_NLTK.json', 'w') as g, open('entities_NLTK.json', 'w') as h:
        json.dump(counts, g)
        for key in resulted_entities:
            resulted_entities[key] = list(resulted_entities[key])
        json.dump(resulted_entities, h)


if __name__ == '__main__':
    # main()
    main2()
