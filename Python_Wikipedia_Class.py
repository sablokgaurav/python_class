# generating and tags words class for the preparation
# using the beautifulSoup
import wikipedia
import re
import os
import requests
from language_detector import detect_language
from langdetect import detect, detect_langs, DetectorFactory
from collections import Counter
from googletranslate import Translator
from nltk.corpus import stopwords
import vaderSentiment.vaderSentiment
import ps4
from bs4 import BeautifulSoup


class GenerateWikipedia:
    def __init__(self, lang, search, summary, page, input_word, input_num):
        self.lang = lang
        self.input_word = str(input_word)
        self.input_num = int(input_num)
        self.page = page
        self.search = wikipedia('self.input_word')
        self.summary = wikipedia('self.input_word', sentences=self.input_num)
        print(f'the_set_language_for_analysis:{self.lang}')
        print(f'the_input_word_for_the_search:{self.word} + {len(self.word)}')
        print(f'the_requested_page_for_the_search:{wikipedia.page(self.page)}')


def detectLangFrequency(self, lang):
    for i in self.input_word:
        if type(self.input_word) == str:
            print(f'the_type_of_input:{str(self.input_word)}')
        DetectorFactory.seed = 0
        language_detect = []
        language_detect.append(detect(self.input_word))
        if language_detect == 'en':
        unique_char = []
        words_frequency = {}
        unique_char.append([j for i in ([j for i in ([list(map(lambda n: n.strip().split(), i))
                        for i in (re.split(r'\n', self.input_word))]) for j in i]) for j in i])
        words_frequency.append(Counter([j for i in ([j for i in ([list(map(lambda n: n.strip().split(), i))
                        for i in (re.split(r'\n', self.input_word))]) for j in i]) for j in i]).most_common(self.input_num))
        print(f'the_frequency_of_the_most_common_word: {words_frequency}')
        if language_detect != 'en':
        translator = Translator()
        translated_tags = translator.translate(self.input_word, lang).self.input_word
        translated_char = [j for i in ([j for i in ([list(map(lambda n: n.strip().split(), i))
                                for i in (re.split(r'\n', self.input_word))]) for j in i]) for j in i]
        translated_words_frequency = Counter([j for i in ([j for i in ([list(map(lambda n: n.strip().split(), i))
                                    for i in (re.split(r'\n', self.input_word))]) for j in i]) 
                                                                         for j in i]).most_common(self.input_num)
        return unique_char, words_frequency, translated_char, translated_words_frequency


def cleanTags(self, tags=None, tagsMake=None):
    if self.word is not None:
        tags = [''.join([j for i in (list(map(lambda n: list(n), [j for i in ([j for i in ([list(map(lambda n:
                                    (n.strip().split()), self.word))]) for j in i]) for j in i]))) for j in i])]
    tagsMake = [[tags[i:i + self.num]
        for i in range(len(tags) - (self.num - 1) + 1)]]
    return tags, tagsMake


def multiPageWords(self):
    multi_page = ['''self.summary''']
    punctuations = [j for i in (
        list(map(lambda n: list(n), ['!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']))) for j in i]
    filter_summary = [[j for i in ([j for i in ([list(filter(lambda n: n != [], i))
                      for i in ([list(map(lambda n: n.split(), i)) for i in multi_page])])for j in i]) for j in i]]
    final_text = [i for i in list(filter_summary)
                  if i not in list(punctuations)]
    return final_text, filter_summary


def nltkStopwords(self, input_query):
    text_analysis = [j for i in ([i.split() for i in self.summary]) for j in i]
    clean_stopwords = set(stopwords.words('english'))
    stop_check = ['input_query', [j for i in (list(
        map(lambda n: list(n), ['!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']))) for j in i]]
    clean_stopwords.update(stop_check)
    for i in stop_check:
        cleaned_text = [list(map(lambda n: n.replace('j', ''), i)) for i in ([j for i in
                                                    ([i.split() for i in text_analysis]) for j in i])]
    return cleaned_text


def sentimentAnalysis(self):
    analysis = SentimentIntensityAnalyzer()
    scores = [analysis.polarity_scores(i)['compound']
              for i in self.cleaned_text]
    final_report = []
    for i in scores:
        if i >= 0.00 and i <= 2.00:
            final_report.append([i, 'Negative'])
        elif i >= 2.00 and i <= 4.00:
            final_report.append([i, 'Neutral'])
        elif i >= 4.00 and i <= 6.00:
            final_report.append([i, 'Positive'])
        else:
            return i
    return final_report

Class BeautifulSoup:
'''This is the implementation of the request library 
    and the beautifulSoup for the scrapping
    and making the tags out of the text. '''
    
def __init__(self, lang, search, summary, page, input_word, input_num):
    self.lang = lang
    self.input_word = str(input_word)
    self.input_num = int(input_num)
    self.page = page
    self.search = wikipedia('self.input_word')
    self.summary = wikipedia('self.input_word', sentences=self.input_num)
    print(f'the_set_language_for_analysis:{self.lang}')
    print(f'the_input_word_for_the_search:{self.word} + {len(self.word)}')
    print(f'the_requested_page_for_the_search:{wikipedia.page(self.page)}')


def detectLangFrequency(self, lang):
    '''This is used to detect the language and if the
     language is english then it will extract the english
     word from the input file.'''
    for i in self.input_word:
        if type(self.input_word) == str:
            print(f'the_type_of_input:{str(self.input_word)}')
        DetectorFactory.seed = 0
        language_detect = []
        language_detect.append(detect(self.input_word))
        if language_detect = = 'en':
        unique_char = []
        words_frequency = {}
        unique_char.append([j for i in ([j for i in ([list(map(lambda n: n.strip().split(), i))
                                for i in (re.split(r'\n', self.input_word))]) for j in i]) for j in i])
        words_frequency.append(Counter([j for i in ([j for i in ([list(map(lambda n: n.strip().split(), i))
                        for i in (re.split(r'\n', self.input_word))]) for j in i]) for j in i]).most_common(self.input_num))
        print(f'the_frequency_of_the_most_common_word: {words_frequency}')
        if language_detect ! = 'en':
        translator = Translator()
        translated_tags = translator.translate(self.input_word, lang).self.input_word
        translated_unique_char = [j for i in ([j for i in ([list(map(lambda n: n.strip().split(), i))
                                           for i in (re.split(r'\n', translated_tags))]) for j in i]) for j in i]
        translated_words_frequency = Counter([j for i in ([j for i in ([list(map(lambda n: n.strip().split(), i))
                                                             for i in (re.split(r'\n', translated_unique_char))]) 
                                                            for j in i]) for j in i]).most_common(self.input_num)
        return unique_char, words_frequency, translated_unique_char, translated_words_frequency


def cleanTags(self, tags=None, tagsMake=None):
   '''This will search the word from the wikipedia and will
      prepare the clean tags, only a single word search allowed '''
   if self.word is not None:
        tags = [''.join([j for i in (list(map(lambda n: list(n), [j for i in ([j for i in ([list(map(lambda n:
                                    (n.strip().split()), self.word))]) for j in i]) for j in i]))) for j in i])]
        tagsMake = [[tags[i:i + self.num] for i in range(len(tags) - (self.num - 1) + 1)]]
        return tags, tagsMake


def multiPageWords(self):
    """
         This will download the required page and then
         will prepare the clean tags from the page.
    """
    multi_page = ['''self.summary''']
    punctuations = [j for i in (
        list(map(lambda n: list(n), ['!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']))) for j in i]
    filter_summary = [[j for i in ([j for i in ([list(filter(lambda n: n != [], i))
                            for i in ([list(map(lambda n: n.split(), i)) for i in multi_page])])for j in i]) for j in i]]
    final_text = [i for i in list(filter_summary) if i not in list(punctuations)]
    return final_text,filter_summary


def nltkStopwords(self, input_query):
    text_analysis = [j for i in ([i.split() for i in self.summary]) for j in i]
    clean_stopwords = set(stopwords.words('english'))
    stop_check = ['input_query', [j for i in (list(map(lambda n: list(n), ['!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']))) for j in i]]
    clean_stopwords.update(stop_check)
    for i in stop_check:
        cleaned_text = [list(map(lambda n: n.replace('j', ''), i)) for i in ([j for i in ([i.split() for i in text_analysis]) for j in i])]
    return cleaned_text


def sentimentAnalysis(self):
    analysis = SentimentIntensityAnalyzer()
    scores = [analysis.polarity_scores(i)['compound']
              for i in self.cleaned_text]
    final_report = []
    for i in scores:
        if i >= 0.00 and i <= 2.00:
            final_report.append([i, 'Negative'])
        elif i >= 2.00 and i <= 4.00:
            final_report.append([i, 'Neutral'])
        elif i >= 4.00 and i <= 6.00:
            final_report.append([i, 'Positive'])
        else:
            return i
    return final_report

def tagReducer(self):
    
    [print(j, len(list(map(lambda n: 0 if n in  [list(''.join({chr(c+65) 
                                    for c in range(26)}).lower())] else 1, j)))) for j in [list(i) for i in a.split()]]
    
def trainingModel(self):
    

        import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text  # Needed for loading universal-sentence-encoder-cmlm/multilingual-preprocess
import numpy as np

def normalization(embeds):
  norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
  return embeds/norms

english_sentences = tf.constant(["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."])
italian_sentences = tf.constant(["cane", "I cuccioli sono carini.", "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane."])
japanese_sentences = tf.constant(["犬", "子犬はいいです", "私は犬と一緒にビーチを散歩するのが好きです"])

preprocessor = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
encoder = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2")

english_embeds = encoder(preprocessor(english_sentences))["default"]
japanese_embeds = encoder(preprocessor(japanese_sentences))["default"]
italian_embeds = encoder(preprocessor(italian_sentences))["default"]

# For semantic similarity tasks, apply l2 normalization to embeddings
english_embeds = normalization(english_embeds)
japanese_embeds = normalization(japanese_embeds)
italian_embeds = normalization(italian_embeds)

print (np.matmul(english_embeds, np.transpose(italian_embeds)))

# English-Japanese similarity
print (np.matmul(english_embeds, np.transpose(japanese_embeds)))

# Italian-Japanese similarity
print (np.matmul(italian_embeds, np.transpose(japanese_embeds)))


