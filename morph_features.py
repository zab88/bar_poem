# -*- coding: utf-8 -*-
import nltk
import string
import pymorphy
from nltk.corpus import stopwords
import os

corpus_dir = 'corpus_1836'
#corpus_dir = 'poems'
morph = pymorphy.get_morph('C:/DB/ru.sqlite-json')
stop_words = stopwords.words('russian')
stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как'])

all_poems = []
all_poems_names = []


def tokenizeMe(sentence):
    all_words = []

    tokens = nltk.word_tokenize(sentence)
    #print(tokens)
    sw = [i.decode('utf-8') for i in stop_words]
    sentence_cleared = [i for i in tokens if ( i not in sw and i not in string.punctuation )]

    part_of_speech = {}
    for word in sentence_cleared:
        word = word.upper()
        info = morph.get_graminfo(word)
        if info:
            all_words.append( info[0]['norm'] )
            #info[0]['class'] # часть речи
            #print info[0]['info'] # инфо
            sp = info[0]['class'].encode('utf-8');
            part_of_speech[sp] = part_of_speech.get(sp, 0) + 1
            if sp == "П":
                #print(info[0]['info'])
                if info[0]['info'][:5] == u"сравн":
                    #print(word)
                    part_of_speech['П_СРАВН'] = part_of_speech.get('П_СРАВН', 0) + 1

    return all_words, part_of_speech

TEMPLATE = [u'П+С', u'ПРИЧАСТИЕ+С', u'С+С']
def getBiMorph(sentence):
    tokens = nltk.word_tokenize(sentence)
    sentence_length = len(tokens)
    found = []
    for k, word in enumerate(tokens):
        #if next word not set
        if sentence_length < (k+2):
            break
        word = word.upper()
        word_next = tokens[k+1].upper()
        #if any of them is punctuation - continue
        if word in string.punctuation or word_next in string.punctuation:
            continue
        info = morph.get_graminfo(word)
        info_next = morph.get_graminfo(word_next)
        if info and info_next:
            if (info[0]['class'] + '+' + info_next[0]['class']) in TEMPLATE:
                print(word)
                print(word_next)
                print('!!!')
                found.append( tuple([info[0]['class'], info_next[0]['class']]) )
    return found

#just for testing
poem = open('poems/4.txt', 'r')
text = poem.read()
text = text.decode('utf-8')
morph_features = getBiMorph(text)
print(morph_features)

#count part of speech
if False:
    for subdir, dirs, files in os.walk(corpus_dir):
        for file in files:
            file_path = subdir + os.path.sep + file
            poem = open(file_path, 'r')
            text = poem.read()
            text = text.decode('utf-8')
            poem.close()

            poem_words, poem_sp = tokenizeMe(text)
            all_poems.append(poem_sp)
            all_poems_names.append(file)

    #print(all_poems)
    out = open('out/out_morph.txt', 'w')
    for ii, poem in enumerate( all_poems ):
        out.write(all_poems_names[ii])
        for k, el in poem.items():
            #print(k)
            #print(el)
            out.write(str(k) + ' - ' + str(el) + "\n")
        out.write("\n")