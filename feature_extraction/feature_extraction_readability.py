# coding='UTF-8'
from bs4 import BeautifulSoup
import random
import time
import numpy as np
import re
import os
import pandas as pd
import csv
import math
from nltk.corpus import stopwords
from nltk import word_tokenize,Text,pos_tag
from nltk.stem import SnowballStemmer


### readability score feature ###
peacock_word_set = {'acclaimed', 'amazing', 'astonishing', 'authoritative', 'beautiful', 'best', 'brilliant', 
                    'can onical', 'celebrated', 'charismatic', 'classic', 'cutting-edged', 'defining', 'definitive', 
                    'eminent', 'enigma', 'exciting', 'extraordinary', 'fabulous', 'famous', 'infamous', 'fantastic', 
                    'fully', 'genius', 'global', 'great', 'greatest', 'iconic', 'immensely', 'impactful', 'incendiary', 
                    'indisputable', 'influential', 'innovative', 'inspired', 'intriguing', 'leader', 'leading', 
                    'legendary', 'major', 'masterly', 'mature', 'memorable', 'notable', 'outstanding', 'pioneer', 
                    'popular', 'prestigious', 'remarkable', 'renowned', 'respected', 'seminal', 
                    'significant', 'skillful', 'solution', 'single-handedly', 'staunch', 'talented',
                    'top', 'transcendent', 'undoubtedly', 'unique', 'visionary', 'virtually', 'virtuoso', 'well-known', 
                    'well-established', 'world-class', 'worst'}

peacock_phrase = { 'the most',
                   'really good'
                    }

weasel_word_set = {  
                   'about', 'adequate', 'and', 'or', 'appropriate', 'approximately', 'basically', 'clearly', 
                   'completely', 'exceedingly', 'excellent', 'extremely', 'fairly', 'few', 'frequently', 'good', 
                   'huge', 'indicated', 'interestingly', 'largely', 'major', 'many', 'maybe', 'mostly', 'normally', 'often',
                   'perhaps', 'primary', 'quite', 'relatively', 'relevant', 'remarkably', 'roughly', 'significantly', 'several', 
                   'sometimes', 'substantially', 'suitable', 'surprisingly', 'tentatively', 'tiny', 'try', 'typically', 'usually', 
                   'valid', 'various', 'vast', 'very'
                   }
weasel_phrase = {  'are a number', 'as applicable', 'as circumstances dictate', 'as much as possible', 'as needed', 'as required', 
                   'as soon as possible', 'at your earliest convenience', 'critics say', 'depending on', 'experts declare', 
                   'if appropriate', 'if required', 'if warranted', 'is a number', 'in a timely manner', 'in general', 'in most cases', 
                   'in our opinion', 'in some cases', 'in most instances', 'it is believed', 'it is often reported', 'it is our understanding', 'it is widely thought', 
                   'it may', 'it was proven', 'make an effort to', 'many are of the opinion', 'many people think', 'more or less', 
                   'most feel', 'on occasion', 'research has shown', 'science says', 'should be', 'some people say', 'striving for', 'we intend to', 
                   'when necessary', 'when possible'
                   }

lines = open('./DaleChallEasyWordList.txt', 'r').readlines()
dale_chall_word_list = set()
for line in lines:
    temp = line.split(' ')
    res = temp[0]
    if temp[0][-1] == '\n':
        res = temp[0][0:-1]
    dale_chall_word_list.add(res)

## Automated readability index 1
def getARI(charCnt, wordCnt, sentCnt):
    ari = 4.71 * charCnt / wordCnt + 0.5 * wordCnt / sentCnt - 21.43
    if ari < 0:
        ari = 0
    return ari

def BormuthIndex(charCnt, wordCnt, sentCnt, diffwordCnt):
    BI = 0.886593 - 0.0364 * charCnt / wordCnt + 0.161911 * (1 - diffwordCnt / wordCnt) - 0.21401 * wordCnt / sentCnt - 0.000577 * wordCnt / sentCnt - 0.000005 * wordCnt / sentCnt
    return BI

def isDifficultWord(word):
    if word in dale_chall_word_list:
        return False
    return True

def isPeacockWord(word):
    if word in peacock_word_set:
        return True
    return False

def hasPeacockPhrase(sentence):
    for phrase in peacock_phrase:
        res = sentence.find(phrase)
        if res != -1:
            #print("peacock")
            #print(phrase)
            return True
    return False

def isWeaselWord(word):
    if word in weasel_word_set:
        return True
    return False

def hasWeaselPhrase(sentence):
    for phrase in weasel_phrase:
        res = sentence.find(phrase)
        if res != -1:
            #print("weasel")
            #print(phrase)
            return True
    return False

def isStopWord(word):
    if word in stopwords.words('english'):
        return True
    return False

## Coleman-Liar index 1
def getColemanLaurIndex(charCnt, wordCnt, sentCnt):
    CLIndex = 5.89 * charCnt / wordCnt - 30.0 * sentCnt / wordCnt - 15.8
    return CLIndex

## FORCAST readability
def getEduYears(oneSyllableCnt):
    return 20 - oneSyllableCnt / 10.0

## Flesch reading ease 1
def getFleschReading(wordCnt, sentCnt, oneSyllableCnt):
    return 206.835 - 1.015 * wordCnt / sentCnt - 84.6 * oneSyllableCnt / wordCnt

## Flesch-Kincaid 1
def getFleschKincaid(wordCnt, sentCnt, oneSyllableCnt):
    return 0.39 * wordCnt / sentCnt + 11.8 * oneSyllableCnt / wordCnt - 15.59

## Gunning Fog Index 1
def getGunningFogIndex(wordCnt, sentCnt, complexWordCnt):
    return 0.4 * ((wordCnt / sentCnt) + 100.0 * (float(complexWordCnt) / wordCnt))

## Lasbarhedsindex 1
def getLIX(wordCnt, sentCnt, longWordCnt):
    return float(wordCnt) / sentCnt + 100 * float(longWordCnt) / wordCnt

## Miyazaki EFL readability index
def getMiyazakiEFL(charCnt, wordCnt, sentCnt):
    return 164.935 - 18.792 * charCnt / wordCnt - 1.916 * wordCnt / sentCnt

## new Dale Chall
def getNewDaleChall(wordCnt, sentCnt, diffwordCnt):
    return 0.1579 * diffwordCnt / wordCnt + 0.0496 * wordCnt / sentCnt

## SMOG grading: 1
def getSmogGrading(sentCnt, complexWordCnt):
    return math.sqrt(30.0 * complexWordCnt / sentCnt) + 3

def get_readability_features(content):
    # parse pure content into sentences and words
    pass

be_set = {
          "is",
          "been",
          "are", 
          "was",
          "were",
          "be"
          }

be_not_set = {
            "isn't",
            "aren't",
            "wasn't",
            "weren't"
            }

do_set = {
          "do", 
          "did",
          "does"
          }

do_not_set = {
            "don't",
            "didn't",
            "doesn't"
            }

have_set = {
            "have",
            "had",
            "has"
            }

prepositions_set_one_word = {
                    "aboard", "about", "above", "after", "against", "alongside", 
                    "amid", "among", "around", "at", "before", "behind", "below", 
                    "beneath", "beside", "besides", "between", "beyond", "but", 
                    "concerning", "considering", "despite", "down", "during", 
                    "except", "inside", "into", "like", "off",
                    "onto", "on", "opposite", "out", "outside", "over",
                    "past", "regarding", "round", "since", 
                    "together", "with", "throughout", "through", 
                    "till", "toward", "under", "underneath", "until", "unto", "up", 
                    "up to", "upon", "with", "within", "without", "across", 
                    "along", "by", "of", "in", "to", "near", "of", "from"
                    }

prepositions_set_two_words = {
                    "according to", "across from", "alongside of", "along with",
                    "apart from", "aside from", "away from", "back of", "because of", 
                    "by means of", "down from", "except for", "excepting for", "from among", 
                    "from between", "from under", "inside of", "instead of", "near to", "out of", 
                    "outside of", "over to", "owing to", "prior to", "round about", "subsequent to"
                    }

prepositions_set_three_words = {
                    "in addition to", "in behalf of", "in spite of",
                    "in front of", "in place of", "in regard to",
                    "on account of", "on behalf of", "on top of"
                    }

subordinate_conjunction_set_one_word = {
                                "after", "because", "lest", "till", "’til", "although", 
                                "before", "unless", "as", "provided", "until", "since",
                                "whenever", "if", "than", "inasmuch", "though", "while"
                               }

subordinate_conjunction_set_two_word = {
                                "now that", "even if", "provided that", 
                                "as if", "even though", "so that"
                               }

subordinate_conjunction_set_three_word = {
                                         "as long as", "as much as", 
                                         "as soon as", "in order that"   
                                         }       

def isToBeWord(word):
    if word in be_set or word in be_not_set:
        return True
    return False
    
def SentenceBeginWithSubConj(sentence):
    sentence = sentence.lower()
    tokens = word_tokenize(sentence)
    if len(tokens) >= 3:
        first_three = tokens[0] + " " + tokens[1] + " " + tokens[2]
        if first_three in prepositions_set_three_words:
            return False
    if len(tokens) >= 2:
        first_two = tokens[0] + " " + tokens[1]
        if first_two in prepositions_set_two_words:
            return False
    for three_words in subordinate_conjunction_set_three_word:
        if sentence.startswith(three_words):
            return True
    for two_words in subordinate_conjunction_set_two_word:
        if sentence.startswith(two_words):
            return True
    if tokens[0] in subordinate_conjunction_set_one_word and tokens[0] not in prepositions_set_one_word:
        return True
    return False

### https://stackoverflow.com/questions/405161/detecting-syllables-in-a-word
def countSyllables(word):
    vowels = "aeiouy"
    numVowels = 0
    lastWasVowel = False
    for wc in word:
        foundVowel = False
        for v in vowels:
            if v == wc:
                if not lastWasVowel: 
                    numVowels += 1   #don't count diphthongs
                    foundVowel = True
                    lastWasVowel = True
                    break
                else:
                    foundVowel = True
                    lastWasVowel = True
                    break
        if not foundVowel:  #If full cycle and no vowel found, set lastWasVowel to false
            lastWasVowel = False
    if len(word) > 2 and word[-2:] == "es": #Remove es - it's "usually" silent (?)
        numVowels-=1
    elif len(word) > 1 and word[-1:] == "e":    #remove silent e
        numVowels-=1
    if numVowels == 0:
        numVowels = 1
    return numVowels

def hasAuxiliaryVerb(sentence):
    tokens = word_tokenize(sentence)
    text = Text(tokens)
    tags = pos_tag(text)
    for index, token in enumerate(tokens):
        # if sentence has "don't", "doesn't", "didn't", it must have ausiliary verb
        if token in do_not_set:
            return True
        
        if token in be_set:
            # are done
            if index != len(tokens) - 1 and tags[index + 1][1] == 'VBN':
                return True
            # are not done
            if index != len(tokens) - 2 and tags[index + 2][1] == 'VBN':
                return True
            # are doing
            if index != len(tokens) - 1 and tags[index + 1][1] == 'VBG':
                return True
            # are not doing
            if index != len(tokens) - 2 and tags[index + 2][1] == 'VBG':
                return True
            
        if token in be_not_set:
            # aren't done
            if index != len(tokens) - 1 and tags[index + 1][1] == 'VBN':
                return True
            # aren't doing
            if index != len(tokens) - 1 and tags[index + 1][1] == 'VBG':
                return True
            
        if token in do_set:
            if index != len(tokens) - 1 and tags[index + 1][1] == 'VB':
                return True
            if index != len(tokens) - 2 and tags[index + 2][1] == 'VB':
                return True
            
        if token in have_set:
            # have done
            if index != len(tokens) - 1 and tags[index + 1][1] == 'VBN':
                # has limited powers XXXXXX
                if index != len(tokens) - 2 and tags[index + 2][1] == 'NN' or index != len(tokens) - 2 and tags[index + 2][1] == 'NNS':
                    return False
                else: 
                    return True
            # have not done
            if index !=len(tokens) - 2 and tags[index + 2][1] == 'VBN':
                return True

    for tag in tags:
        if tag[0] == 'MD':
            return True
    return False
    
def getConjunctionCount(sentence):
    conjunction_number = 0
    tokens = word_tokenize(sentence)
    text = Text(tokens)
    tags = pos_tag(text)  
    for tag in tags:
        if tag[1] == "CC" or tag[1] == "IN":
            conjunction_number = conjunction_number + 1
    return conjunction_number

def getPrepositionCount(sentence):
    preposition_number = 0
    tokens = word_tokenize(sentence)
    text = Text(tokens)
    tags = pos_tag(text)  
    for tag in tags:
        if tag[1] == "IN":
            preposition_number = preposition_number + 1
    return preposition_number

def getPronounCount(sentence):
    pronoun_number = 0
    tokens = word_tokenize(sentence)
    text = Text(tokens)
    tags = pos_tag(text)
    for tag in tags:
        if tag[1] == "PRP" or tag[1] == "PRP$":
            pronoun_number = pronoun_number + 1
    return pronoun_number
        
def SentenceBeginWithConj(sentence):
    tokens = word_tokenize(sentence)
    text = Text(tokens)
    tags = pos_tag(text)
    if tags[0][1] == "CC" or tags[0][1] == "IN":
        return True
    return False

def SentenceBeginWithPrep(sentence):
    tokens = word_tokenize(sentence)
    text = Text(tokens)
    tags = pos_tag(text)
    if tags[0][1] == "IN" and tags[0][0].lower() not in subordinate_conjunction_set_one_word:
        return True
    return False

def SentenceBeginWithInterrogativePronoun(sentence):
    tokens = word_tokenize(sentence)
    interrogative_pronoun = {'What', 'Which', 'Who', 'Whom', 
                             'Whose', 'Whatever', 'Whatsoever',
                             'Whichever','Whoever','Whosoever', 
                             'Whomever', 'Whomsoever', 'Whosever',
                             'Why', 'where', 'When', 'How'
                             }
    if tokens[0] in interrogative_pronoun:
        return True
    return False

def SentenceBeginWithPronoun(sentence):
    tokens = word_tokenize(sentence)
    text = Text(tokens)
    tags = pos_tag(text)
    if tags[0][1] == "PRP" or tags[0][1] == "PRP$":
        return True
    return False 

def isWordNominalization(word):
    if len(word) <= 4:
        return False
    suffix = word[-4:]
    nominalization_suffix = {'tion', 'ment', 'ence', 'ance'}
    snow = SnowballStemmer('english') 
    stem = snow.stem(word)
    # check whether it is suffix. For example, 'ance' in France is not suffix. 
    if suffix in nominalization_suffix and len(word) - len(stem) >= 3:
        return True
    return False

def SentencePassiveVoice(sentence):
    tokens = word_tokenize(sentence)
    text = Text(tokens)
    tags = pos_tag(text) 
    for index, token in enumerate(tokens):
        if token in be_set:
            # are done
            if index < len(tokens) - 1 and tags[index + 1][1] == 'VBN':
                return True
            # are not done
            if index < len(tokens) - 2 and tags[index + 2][1] == 'VBN':
                return True
            # are not properly done
            if index < len(tokens) - 3 and tags[index + 3][1] == 'VBN':
                return True
        if token in be_not_set:
            # isn't done
            if index < len(tokens) - 1 and tags[index + 1][1] == 'VBN':
                return True
            # isn't properly done
            if index < len(tokens) - 2 and tags[index + 2][1] == 'VBN':
                return True
    return False


def get_text_stat_features(content):
    pass

def get_basic_stats_data(content):
    word_cnt = 0
    char_cnt = 0
    sentence_cnt = 0
    max_sentence_words_len = 0
    paragraph_cnt = 1
    abstract_chars_len = 0
    abstract_words_len = 0
    std_dev_sections_chars_len = 0
    std_dev_sections_words_len = 0
    max_section_chars_len = 0
    max_section_words_len = 0
    min_section_chars_len = 0
    min_section_words_len = 0

    word_content = content.replace('-', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('&', '')
    word_list = re.split(r"[|\n|\s|,|.|!|]", word_content)
    word_list = [re.sub('[^a-zA-Z]','',item) for item in word_list if len(item) > 0]
    # print('Word list :', word_list)
    word_cnt = len(word_list)

    char_list = content.replace(' ', '')
    char_cnt = len(char_list)

    sentence_list = re.split(r"[|\n|?|.|!|]", content)
    sentence_list = [item for item in sentence_list if len(item) > 0 and item != ' ' and '[edit]' not in item]
    # print('Sentence list :', sentence_list)
    sentence_cnt = len(sentence_list)

    oneSyllableCnt = 0
    complexWordCnt = 0
    longWordCnt = 0
    for word in word_list:
        word = word.lower()
        if countSyllables(word) == 1:
            oneSyllableCnt += 1
        if isDifficultWord(word):
            complexWordCnt += 1
        if len(word) >= 7:
            longWordCnt += 1
    
    readability_features = []

    ARI = getARI(char_cnt, word_cnt, sentence_cnt)
    coleman_liau = getColemanLaurIndex(char_cnt, word_cnt, sentence_cnt)
    flesh = getFleschReading(char_cnt, word_cnt, oneSyllableCnt)
    Kincaid = getFleschKincaid(char_cnt, word_cnt, oneSyllableCnt)
    fog = getGunningFogIndex(char_cnt, word_cnt, complexWordCnt)
    Lix = getLIX(word_cnt, sentence_cnt, longWordCnt)
    Smog = getSmogGrading(sentence_cnt, complexWordCnt)

    readability_features.append(ARI)
    readability_features.append(coleman_liau)
    readability_features.append(flesh)
    readability_features.append(Kincaid)
    readability_features.append(fog)
    readability_features.append(Lix)
    readability_features.append(Smog)

    return  word_list, sentence_list, readability_features


if __name__ == '__main__':
    file_dir = './data/pages/plain_txt_name_index/'
    files = os.listdir(file_dir)

    readability_feature_file_name = './feature_readbility20200214_sample1000.csv'

    if not os.path.exists(readability_feature_file_name):
        with open(readability_feature_file_name, 'w', newline='', encoding='utf-8-sig') as t_file:
            csv_writer = csv.writer(t_file)
            csv_writer.writerow(('file_name', 'ARI', 'coleman_liau', 'flesh', 'Kincaid','fog', 'Lix', 'Smog'))   

    progress_counter = 0
    for file in files[:1000]:
        with open(file_dir + file, 'r', encoding='UTF-8') as f:
            content = f.read()
        word_list, sentence_list, readability_features = get_basic_stats_data(content)
        with open(readability_feature_file_name, 'a', newline='', encoding='utf-8-sig') as t_file:
            csv_writer = csv.writer(t_file)
            csv_writer.writerow((file, readability_features[0], readability_features[1], 
                                 readability_features[2], readability_features[3],
                                 readability_features[4], readability_features[5], 
                                 readability_features[6]))
        # print(readability_features)
        if progress_counter % 1000 == 0:
            print('Progress comes to: ', float(progress_counter) / float(len(files)))

        progress_counter += 1

    print('Mission finished!!!')
        




    

