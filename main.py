# -*- coding: utf-8 -*-
import nltk
import string
import pymorphy
#from pymorphy.contrib import tokenizers
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os

morph = pymorphy.get_morph('C:/DB/ru.sqlite-json')
stop_words = stopwords.words('russian')
stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как'])
corpus = {}

def tokenizeMe(sentence):
    all_words = []
    #sentence = "меня! это,. что? похоже; на - пирамиду"
    #sentence = sentence.encode('utf-8').upper()
    #sentence = sentence.upper()
    #print(sentence)
    #
    #words = tokenizers.extract_words(sentence)
    #print(words)

    tokens = nltk.word_tokenize(sentence)
    #print(tokens)
    sw = [i.decode('utf-8') for i in stop_words]
    sentence_cleared = [i for i in tokens if ( i not in sw and i not in string.punctuation )]

    for word in sentence_cleared:
        #word.decode('utf8')
        #word = word.decode('utf-8').upper()
        word = word.upper()
        #print(word)
        #word = word.encode('utf-8')
        info = morph.get_graminfo(word)
        if info:
            #print( info[0]['norm'] )
            all_words.append( info[0]['norm'] )
    return all_words


#walk through files
for subdir, dirs, files in os.walk('poems'):
    for file in files:
        file_path = subdir + os.path.sep + file
        poem = open(file_path, 'r')
        text = poem.read()

        #sub_corpus = tokenizeMe(text)
        #corpus[file] = sub_corpus
        corpus[file] = text
        #print(sub_corpus[0])

tfidf = TfidfVectorizer(tokenizer=tokenizeMe, stop_words=None)
tt = tfidf.fit_transform(corpus.values())

test_file = open('test/1.txt', 'r')
#test_file = open('poems/1.txt', 'r')
text = test_file.read()
response = tfidf.transform([text])
print(response)

feature_names = tfidf.get_feature_names()
for col in response.nonzero()[1]:
    print(feature_names[col])
    print(feature_names[col], ' - ', response[0, col])
