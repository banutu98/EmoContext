from utils import mapping_dict
from utils import parse_file
import re
import string


def word_normalize(conversation):
    result = list()
    for sentence in conversation:
        new_sentence = re.sub("tha+nks ", ' thanks ', sentence.lower())
        new_sentence = re.sub("Tha+nks ", ' Thanks ', new_sentence.lower())
        new_sentence = re.sub("yes+ ", ' yes ', new_sentence.lower())
        new_sentence = re.sub("Yes+ ", ' Yes ', new_sentence)
        new_sentence = re.sub("very+ ", ' very ', new_sentence)
        new_sentence = re.sub("go+d ", ' good ', new_sentence)
        new_sentence = re.sub("Very+ ", ' Very ', new_sentence)
        new_sentence = re.sub("why+ ", ' why ', new_sentence.lower())
        new_sentence = re.sub("wha+t ", ' what ', new_sentence)
        new_sentence = re.sub("sil+y ", ' silly ', new_sentence)
        new_sentence = re.sub("hm+ ", ' hmm ', new_sentence)
        new_sentence = re.sub(" no+ ", ' no ', ' ' + new_sentence)
        new_sentence = re.sub("sor+y ", ' sorry ', new_sentence)
        new_sentence = re.sub("so+ ", ' so ', new_sentence)
        new_sentence = re.sub("lie+ ", ' lie ', new_sentence)
        new_sentence = re.sub("okay+ ", ' okay ', new_sentence)
        new_sentence = re.sub(' lol[a-z]+ ', ' laugh out loud ', new_sentence)
        new_sentence = re.sub(' wow+ ', ' wow ', new_sentence)
        new_sentence = re.sub('wha+ ', ' what ', new_sentence)
        new_sentence = re.sub(' ok[a-z]+ ', ' ok ', new_sentence)
        new_sentence = re.sub(' u+ ', ' you ', new_sentence)
        new_sentence = re.sub(' wellso+n ', ' well soon ', new_sentence)
        new_sentence = re.sub(' byy+ ', ' bye ', new_sentence.lower())
        new_sentence = re.sub(' ok+ ', ' ok ', new_sentence.lower())
        new_sentence = re.sub('o+h', ' oh ', new_sentence)
        new_sentence = re.sub('you+ ', ' you ', new_sentence)
        new_sentence = re.sub('plz+', ' please ', new_sentence.lower())
        new_sentence = new_sentence.replace('’', '\'').replace('"', ' ').replace("`", "'")
        new_sentence = new_sentence.replace('fuuuuuuukkkhhhhh', 'fuck')
        new_sentence = new_sentence.replace('whats ', 'what is ').replace("what's ", 'what is ').replace("i'm ",
                                                                                                         'i am ')
        new_sentence = new_sentence.replace("it's ", 'it is ')
        new_sentence = new_sentence.replace('Iam ', 'I am ').replace(' iam ', ' i am ').replace(' dnt ', ' do not ')
        new_sentence = new_sentence.replace('I ve ', 'I have ').replace('I m ', ' I am ').replace('i m ', 'i am ')
        new_sentence = new_sentence.replace('Iam ', 'I am ').replace('iam ', 'i am ')
        new_sentence = new_sentence.replace('dont ', 'do not ').replace('google.co.in ', ' google ').replace(' hve ',
                                                                                                             ' have ')
        new_sentence = new_sentence.replace('Ain\'t ', ' are not ').replace(' lv ', ' love ')
        new_sentence = new_sentence.replace(' ok~~ay~~ ', ' okay ').replace(' Its ', ' It is').replace(' its ',
                                                                                                       ' it is ')
        new_sentence = new_sentence.replace('  Nd  ', ' and ').replace(' nd ', ' and ').replace('i ll ', 'i will ')
        new_sentence = new_sentence.replace(" I'd ", ' i would ').replace('&apos;', "'")
        new_sentence = new_sentence.replace(" won ' t ", ' will not ').replace(' aint ', ' am not ')
        result.append(new_sentence)
    return result


def emoticons_normalize(conversation):
    result = list()
    for sentence in conversation:
        for item, value in mapping_dict.EMOTICONS_UNICODE.items():
            if item in sentence:
                sentence = sentence.replace(item, f' {value} ')
        for item, value in mapping_dict.EMOTICONS_TOKEN.items():
            if item in sentence:
                sentence = sentence.replace(item, value)
        sentence = sentence.replace('  ', ' ').strip()
        result.append(sentence)
    return result


def correct_words(conversation):
    result = list()
    for sentence in conversation:
        for word in sentence.split(' '):
            new_word = word.strip(string.punctuation)
            if new_word in mapping_dict.CORRECTIONS:
                corrected_word = mapping_dict.CORRECTIONS[new_word]
                sentence = re.sub(new_word, corrected_word, sentence)
        result.append(sentence)
    return result


def get_normalized_conversations(conversations):
    new_conversations = list()
    for c in range(len(conversations)):
        new_conversation = emoticons_normalize(word_normalize(correct_words(conversations[c])))
        new_conversations.append(new_conversation)
    return new_conversations


def conversations_to_sentences(conversations):
    return [[' <eos> '.join(c)] for c in conversations]


def get_normalized_labels(labels):
    result = list()
    for label in labels:
        if label == 'happy':
            result.append(0)
        elif label == 'sad':
            result.append(1)
        elif label == 'angry':
            result.append(2)
        else:
            result.append(3)
    return result


def write_normalized_conversations_to_file(output_file):
    conversations = parse_file.get_sentences('input_data/train.txt')
    labels = parse_file.get_labels('input_data/train.txt')
    normalized_conversations = get_normalized_conversations(conversations)
    sentences = conversations_to_sentences(normalized_conversations)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in range(len(sentences)):
            f.write(sentences[sentence][0] + '\t' + labels[sentence] + '\n')


if __name__ == '__main__':
    write_normalized_conversations_to_file('input_data/normalized_train_conversations.txt')
