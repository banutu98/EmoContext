from utils.parse_file import get_sentences
from textblob import TextBlob
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def nltk_method():
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    conversations = get_sentences(r"../train.txt")
    with open("sentiment_analysis.txt", 'w', encoding='utf-8') as f:
        for conversation in conversations:
            for sentence in conversation:
                print(sentence, file=f)
                ss = sid.polarity_scores(sentence)
                for k in sorted(ss):
                    print('{0}: {1}, '.format(k, ss[k]), end='', file=f)
                print(file=f)


def text_blob_method():
    conversations = get_sentences(r"../train.txt")
    sentiments = dict()
    sentiments["positive"] = list()
    sentiments["negative"] = list()
    sentiments["neutral"] = list()
    for conversation in conversations:
        polarity = 0
        for sentence in conversation:
            polarity += TextBlob(sentence).sentiment[0]
        polarity /= 3
        if -0.1 <= polarity <= 0.1:
            sentiments["neutral"].append(conversation)
        elif polarity > 0.1:
            sentiments["positive"].append(conversation)
        else:
            sentiments["negative"].append(conversation)
    with open("sentiments.json", 'w', encoding='utf-8') as f:
        print(json.dumps(sentiments, ensure_ascii=False, indent=3), file=f)
    with open("statistics_result_NLTK.json", 'r') as f:
        stats = json.load(f)
    with open("statistics_result_NLTK.json", 'w') as f:
        stats["Total number of analyzed conversations"] = len(sentiments["neutral"]) + len(
            sentiments["positive"]) + len(sentiments["negative"])
        stats["Positive conversations"] = len(sentiments["positive"])
        stats["Negative conversations"] = len(sentiments["negative"])
        stats["Neutral conversations"] = len(sentiments["neutral"])
        print(json.dumps(stats, indent=3), file=f)


if __name__ == '__main__':
    text_blob_method()
    nltk_method()
