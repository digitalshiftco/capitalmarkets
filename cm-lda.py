import re
import gc
import numpy as np
import pandas as pd
from pprint import pprint

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import datapath

import spacy
import pyLDAvis
import pyLDAvis.gensim

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# read training data
print "reading training data"
df = pd.read_csv("/home/data/Projects/data/training_data.csv")

# slice by date
print "using last 3 years"
dft = df[(df['date'] > '2016-12-31') & (df['date'] < '2019-12-31')]

# convert transcript entries to block of texts
print "converting transcripts to list"
data = dft.transcript.values.tolist()

# remove new line characters
print "removing new lines"
data = [re.sub(r'\s+', ' ', token) for token in data]

# remove single quotes
print "removing single quotes"
data = [re.sub("\'", "", token) for token in data]

# preprocess/clean text to lowercase, remove punctuation
print "preprocessing to lowercase and remove punctuation"
def convert_to_words(text):
    for sentence in text:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

data_words = list(convert_to_words(data))

# introduce NLTK
print "extending stopwords"
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend([u'sir', u'from', u'edu', u'subject', u'do', u'have', u'big', u'small', u'low', u'high', u'none', u'may', u'among', u'within', u'don', u't', u'day', u'etc', u'around', u'frequent', u'including', u'even', u'can', u'likely', u'will', u'like', u'today', u'bit', u'put', u'aim', u's', u'got', u'really', u'huge', u'see', u'almost', u'already', u'much', u'recent', u'many', u'change', u'changes', u'someone', u'said', u'says', u'gives', u'give', u'people', u'new', u'say', u'least', u'first', u'last', u'second', u'one', u'two', u'go', u'goes', u'take', u'going', u'taking', u'just', u'can', u'cannot', u'keep', u'keeps', u'also', u'done', u'good', u'get', u'without', u'told', u'might', u'time', u'unable', u'able', u'know', u'end', u'now', u'want', u'didn', u'back', u'doesn', u'couldn', u'since', u'shouldn', u'seen', u'works', u'zero', u'every', u'each', u'other', u'ever', u'neither', u'll', u'mr', u'ms', u'mrs', u'think', u'tomorrow', u'way', u'still', u'know', u'later', u'fine', u'let', u'went', u'night', u've', u'must', u'act', u're', u'c', u'b', u'a', u'done', u'began', u'ones', u'm', u'soon', u'word', u'along', u'main', u'q', u'lot', u'e', u'd', u'entire', u'year', u'mean', u'means', u'important', u'always', u'something', u'rather', u'either', u'makes', u'make', u'uses', u'use', u'enough', u'w', u'd', u'never', u'giving', u'o', u'involve', u'involves', u'involving', u'little', u'inside', u'sat', u'third', u'fourth', u'fifth', u'sixth', u'next', u'given', u'million', u'billion', u'millions', u'billions', u'option', u'options', u'full', u'complete', u'need', u'needs', u'set', u'manage', u'sets', u'manages', u'bring', u'brings', u'brought', u'try', u'tries', u'tried', u'week', u'former', u'monday', u'tuesday', u'wednesday', u'thursday', u'friday', u'saturday', u'sunday', u'spent', u'spend', u'spends', u'month', u'months', u'send', u'sends', u'sent', u'went', u'january', u'february', u'march', u'april', u'may', u'june', u'july', u'august', u'september', u'october', u'november', u'december', u'allow', u'process', u'old', u'times', u'nearly', u'looking', u'looks', u'look', u'thinly', u'becoming', u'stay', u'stays', u'took', u'takes', u'take', u'types', u'type', u'thought', u'though', u'idea', u'clear', u'clearly', u'behind', u'half', u'us', u'less', u'claim', u'claims', u'long', u'short', u'smaller', u'larger', u'bigger', u'largest', u'biggest', u'smallest', u'longer', u'shorter', u'short', u'long', u'extreme', u'severe', u'largely', u'anymore', u'years', u'spoke', u'give', u'gave', u'given', u'gives', u'reportedly', u'supposedly', u'alledgedly', u'please', u'received', u'receive', u'receives', u'longtime', u'best', u'existing', u'putting', u'put', u'puts', u'whose', u'yesterday', u'thing', u'week', u'another', u'month', u'day', u'come', u'would', u'kind'])

# functions for stopwords, bigrams, and lemmatization
def remove_stopwords(text):
    return [[word for word in doc if word not in stop_words] for doc in text]

def make_bigrams(text):
    return [bigram_mod[doc] for doc in text]

def lemmatization(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    output = []
    for sentence in text:
        doc = nlp(" ".join(sentence))
        output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return output

# remove stopwords
print "removing stopwords"
data_words_nostopwords = remove_stopwords(data_words)

# build the bigram model
print "building bigram model"
bigram = gensim.models.Phrases(data_words_nostopwords, min_count=5, threshold=5)

# make faster bigram model
print "building faster bigram model"
bigram_mod = gensim.models.phrases.Phraser(bigram)

#  make bigrams
print "making bigrams"
data_words_bigrams = make_bigrams(data_words_nostopwords)

# initialize spacy english model
nlp = spacy.load('en', disable=['parser', 'ner'])

# lemmatize with only noun, adjective, verb, adverb
print "lemmatizing bigrams with spacy"
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# create dictionary input for LDA
print "creating dictionary"
dictionary = corpora.Dictionary(data_lemmatized)

# filter out rare and common words
print "filtering dictionary"
dictionary.filter_extremes(no_below=20, no_above=0.5)

# create corpus input for LDA using term document frequency
print "creating corpus"
corpus = [dictionary.doc2bow(text) for text in data_lemmatized]

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print "generating model with topic number = " + str(num_topics)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=num_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
        pprint(model.show_topics(formatted=False))
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# compute optimal number of topics
limit=25; start=5; step=5;
x = range(start, limit, step)

model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=data_lemmatized, start=start, limit=limit, step=step)

# print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics = ", m, " has Coherence Value of", round(cv, 4))
