import pandas as pd
import math


def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bowCount)
    return tfDict


def computeIDF(docList):
    idfDict = {}
    N = len(docList)

    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                if word in idfDict:
                    idfDict[word] += 1
                else:
                    idfDict[word] = 1

    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf


if __name__ == '__main__':
    docA = open("dev.txt", 'r', encoding="utf-8").read()
    docB = open("train.txt", 'r', encoding="utf-8").read()
    bowA = docA.split(" ")
    bowB = docB.split(" ")
    wordSet = set(bowA).union(set(bowB))
    wordDictA = dict.fromkeys(wordSet, 0)
    wordDictB = dict.fromkeys(wordSet, 0)
    for word in bowA:
        wordDictA[word] += 1

    for word in bowB:
        wordDictB[word] += 1

    tfBowA = computeTF(wordDictA, bowA)
    tfBowB = computeTF(wordDictB, bowB)
    idfs = computeIDF([wordDictA, wordDictB])

    tfidfBowA = computeTFIDF(tfBowA, idfs)
    tfidfBowB = computeTFIDF(tfBowB, idfs)
    with open("result.txt", 'w', encoding="utf-8") as f:
        print(pd.DataFrame([tfidfBowA, tfidfBowB]).to_string(), file=f)
