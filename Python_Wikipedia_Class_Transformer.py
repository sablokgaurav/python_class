# combining SBERT with the online search from wikipedia and beautiful soup parsing. 
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer, util
import wikipedia
import re
import os
import argparse
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
parser=argparse.ArgumentParser()
parser.add_argument('--word',action='store_true',type=str,help="provide_the_word_for_search")
parser.add_argument('--phrase', action='store_true',type=str,help="provide_the_phrase_for_search")
parser.add_argument('--file_path',action='store_true',type=path(), help="provide_the_file_path")
parser.add_argument('model',action='store',type=str, help='available_models_are:'
                                                 'all-mpnet-base-v2','multi-qa-mpnet-base-dot-v1',
                                                 'all-distilroberta-v1','all-MiniLM-L12-v2',
                                                 'multi-qa-distilbert-cos-v1','all-MiniLM-L6-v2',
                                                 'multi-qa-MiniLM-L6-cos-v1',
                                                 'paraphrase-multilingual-mpnet-base-v2',
                                                 'paraphrase-albert-small-v2',
                                                 'paraphrase-multilingual-MiniLM-L12-v2',
                                                 'paraphrase-MiniLM-L3-v2',
                                                 'distiluse-base-multilingual-cased-v1',
                                                 'distiluse-base-multilingual-cased-v2 ')
args = parser.parse_args

class ModelText:
    """_summary_
    generating the necessary files and the words and the
    phrases
    """


def __init__(self, lang, search:str, summary:str, page, input_word:str, input_num:int):
        self.lang = lang
        self.input_word = str(input_word)
        self.input_num = int(input_num)
        self.page = page
        self.search = wikipedia('self.input_word')
        self.summary = wikipedia('self.input_word', sentences=self.input_num)
        print(f'the_set_language_for_analysis:{self.lang}')
        print(f'the_input_word_for_the_search:{self.word} + {len(self.word)}')
        print(f'the_requested_page_for_the_search:{wikipedia.page(self.page)}')


def languageSimilarity(self, additionalSearchWord:str, model:str):
     """_summary_
          this will detect the similarity search based on the NLI model
           of the SBERT.
        Args:
        additionalSearchWord (_word_): _search_this_word_
        model (_select_the_model_): _select_the_model_from_the_following_
        ['all-mpnet-base-v2','multi-qa-mpnet-base-dot-v1',
            'all-distilroberta-v1','all-MiniLM-L12-v2',
            'multi-qa-distilbert-cos-v1','all-MiniLM-L6-v2',
            'multi-qa-MiniLM-L6-cos-v1',
            'paraphrase-multilingual-mpnet-base-v2',
            'paraphrase-albert-small-v2',
            'paraphrase-multilingual-MiniLM-L12-v2',
            'paraphrase-MiniLM-L3-v2',
            'distiluse-base-multilingual-cased-v1',
            'distiluse-base-multilingual-cased-v2 ']
    """


self.additionalSearch = wikipedia.search(self.additionalSearch)
language_tasking = [tuple('self.search', 'self.additional_search')]
model_encoding = ['all-mpnet-base-v2', 'multi-qa-mpnet-base-dot-v1',
                  'all-distilroberta-v1', 'all-MiniLM-L12-v2',
                  'multi-qa-distilbert-cos-v1', 'all-MiniLM-L6-v2', 'multi-qa-MiniLM-L6-cos-v1',
                  'paraphrase-multilingual-mpnet-base-v2', 'paraphrase-albert-small-v2',
                  'paraphrase-multilingual-MiniLM-L12-v2', 'paraphrase-MiniLM-L3-v2',
                  'distiluse-base-multilingual-cased-v1',
                  'distiluse-base-multilingual-cased-v2 ']
training_model = [i for i in model_encoding if i == model]
prediction_scores = training_model.predict(language_tasking)
labels = ['contradiction', 'entailment', 'neutral']
models_labels = [labels[i] for i in prediction_scores.argmax(axis=1)]
sequence_model = SentenceTransformer(all-MiniLM-L6-v2)
encode_1 = sequence_model.encode(str(self.summary).rsplit().split())
encode_2 = sequence_model.encode(str(self.additionalSearchPhrase))
similarity = [util.cos_sim(encode_1, encode_2)]
print(f'the_similarity_score_is:{i for i in similarity}')
print(
    f'the_predefined_model_for_the_sequence_similarity_search:{str(all-MiniLM-L6-v2)}')


def languageSimilarityPhrases(self, additionalSearchPhrase:str, model:str):
       """_summary_
       this will detect the similarity search based on the NLI model
       of the SBERT.
        Args:
        additionalSearchWord (_word_): _search_this_word_
        model (_select_the_model_): _select_the_model_from_the_following_
        ['all-mpnet-base-v2','multi-qa-mpnet-base-dot-v1',
            'all-distilroberta-v1','all-MiniLM-L12-v2',
            'multi-qa-distilbert-cos-v1','all-MiniLM-L6-v2','multi-qa-MiniLM-L6-cos-v1',
            'paraphrase-multilingual-mpnet-base-v2','paraphrase-albert-small-v2',
            'paraphrase-multilingual-MiniLM-L12-v2','paraphrase-MiniLM-L3-v2',
            'distiluse-base-multilingual-cased-v1',
            'distiluse-base-multilingual-cased-v2 ']
    """


self.additionalSearch = wikipedia.summary(self.additionalSearchPhrase)
language_tasking = [tuple('self.summary', 'self.additional_search')]
model_encoding = ['all-mpnet-base-v2', 'multi-qa-mpnet-base-dot-v1',
                  'all-distilroberta-v1', 'all-MiniLM-L12-v2',
                  'multi-qa-distilbert-cos-v1', 'all-MiniLM-L6-v2', 'multi-qa-MiniLM-L6-cos-v1',
                  'paraphrase-multilingual-mpnet-base-v2', 'paraphrase-albert-small-v2',
                  'paraphrase-multilingual-MiniLM-L12-v2', 'paraphrase-MiniLM-L3-v2',
                  'distiluse-base-multilingual-cased-v1',
                  'distiluse-base-multilingual-cased-v2 ']
training_model = [i for i in model_encoding if i == model]
prediction_scores = training_model.predict(language_tasking)
labels = ['contradiction', 'entailment', 'neutral']
models_labels = [labels[i] for i in prediction_scores.argmax(axis=1)]
encode_1 = sequence_model.encode(str(self.summary).rsplit().split())
encode_2 = sequence_model.encode(str(self.additionalSearchPhrase))
similarity = [util.cos_sim(encode_1, encode_2)]
print(f'the_similarity_score_is:{i for i in similarity}')
print(f'the_model_labels:{model_labels}')


def CombinedSearch(self, additionalWord: str, additionalPhrase: str, model: str | int, search_num: int):
    """_summary_
        in this i combined the wikipedia search and the SBERT
        for the semantic search to enhance the tag similarity
       and i joined the CrossEncoder and the search to build
       a bigger corpus.
       Args:
           additionalWord (_str_): _description_
           Define a additionalWord to search from the pre-trained
           model
           additionalPhrase (_str_): _description_
           Define an additionalPhrase to search from the pre-trained
           model
       model (_select_the_model_): _select_the_model_from_the_following_
        ['all-mpnet-base-v2','multi-qa-mpnet-base-dot-v1',
            'all-distilroberta-v1','all-MiniLM-L12-v2',
            'multi-qa-distilbert-cos-v1','all-MiniLM-L6-v2',
            'multi-qa-MiniLM-L6-cos-v1',
            'paraphrase-multilingual-mpnet-base-v2',
            'paraphrase-albert-small-v2',
            'paraphrase-multilingual-MiniLM-L12-v2',
            'paraphrase-MiniLM-L3-v2',
            'distiluse-base-multilingual-cased-v1',
            'distiluse-base-multilingual-cased-v2 ']
       """

    search_num = [1, 2]
    for i in search_num:
        if i == 1:
               model = Sentencetransformer('all-MiniLM-L6-v2')
               word = [str(self.additionalWord)]
               wiki_word = wikipedia.search(self.additionalWord)
               para = util.paraphrase_mining(model, word)
               final_text_classification = [str(wiki_word), paraphrases]
               training_model = [i for i in model_encoding if i == model]
               prediction_scores = training_model.predict(language_tasking)
               labels = ['contradiction', 'entailment', 'neutral']
               models_labels = [labels[i]
                                for i in prediction_scores.argmax(axis=1)]
               encode = model.encode(final_text_classification)
               similarity_score = util.cos_sim(encode, encode)
               print(f'the_similarity_score:{similarity_score}')
        if i == 2:
               model = Sentencetransformer('all-MiniLM-L6-v2')
               word = [str(self.additionalPhrase)]
               wiki_phrase = wikipedia.summary(self.additionalPhrase)
               para = util.paraphrase_mining(model, wiki_phrase)
               final_text_classification = [str(wiki_phrase), paraphrases]
               training_model = [i for i in model_encoding if i == model]
               prediction_scores = training_model.predict(language_tasking)
               labels = ['contradiction', 'entailment', 'neutral']
               models_labels = [labels[i]
                                for i in prediction_scores.argmax(axis=1)]
               encode = model.encode(final_text_classification)
               similarity_score = util.cos_sim(encode, encode)
               print(f'the_similarity_score:{similarity_score}')


def fileReadCrossEncoders(self, file_path):
    """_summary_
       _provide the
        the file path and it will read the file
        contents and will prepare the clean files
        and prepare it for the cross encoders
        and use the final file for the training
        model_
    Args:
        file_path (_file_path_): 
    """


def fileReadWordsLength(self,) -> List[str]:
    file_content = []
    file_content_length = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if line.startswith('\n'):
                continue
            else:
                file_content.append(line)
        for i in file_contents:
           print(f'the_length_of_the_file_content:{len(i)}')
        return file_content, file_content_length
    for i in file_content:
        search=[]
        search.append(wiki.search(i))
        


      
           
# generating and tags words class for the preparation
# using the beautifulSoup



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


