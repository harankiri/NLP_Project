import nltk
import pandas as pd
import string
import ast
import re

# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")

from nltk.corpus import stopwords
from nltk import sent_tokenize
from nltk import ngrams
from nltk import word_tokenize

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

stop = stopwords.words('english')
tokenize = nltk.word_tokenize
lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
pd.set_option('max_rows', 50)


path = 'data/SMSSpamCollection.tsv'
features = ['label', 'message']
sms = pd.read_table(path, header=None, names=features)
sms.head()

sms['cleaned'] = sms['message'].str.replace(r'[^\w\s]+', '')

sms['lower'] = sms['cleaned'].apply(lambda row:row.lower())

sms['stopword'] = sms['lower'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

sms['tokenizer'] = sms['stopword'].apply(lambda row:tokenize(row))

sms['lemmatizer'] = sms['tokenizer'].apply(lambda row:[lemmatizer.lemmatize(w) for w in row])

sms['bigram'] = sms['lemmatizer'].apply(lambda row:list(ngrams(row, 2)))

bigram_set = sms.groupby('label').agg({'bigram': 'sum'})


spam = {}
for gram in bigram_set.iat[1, 0]:
    if gram not in spam:
        spam[gram] = 1
    else:
        spam[gram] += 1

ham = {}
for gram in bigram_set.iat[0, 0]:
    if gram not in ham:
        ham[gram] = 1
    else:
        ham[gram] += 1


path = 'data/SMSSpamCollection - Copy.tsv'
features = ['message']
sentence = pd.read_table(path, header=None, names=features)

sentence['cleaned'] = sentence['message'].str.replace(r'[^\w\s]+', '')

sentence['lower'] = sentence['cleaned'].apply(lambda row:row.lower())

sentence['stopword'] = sentence['lower'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

sentence['tokenizer'] = sentence['stopword'].apply(lambda row:tokenize(row))

sentence['lemmatizer'] = sentence['tokenizer'].apply(lambda row:[lemmatizer.lemmatize(w) for w in row])

sentence['bigram'] = sentence['lemmatizer'].apply(lambda row:list(ngrams(row, 2)))


def hbprob(w, w1):
    try:
        a = ham[w]
    except KeyError:
        a = 0

    try:
        ab = ham[w, w1]
    except KeyError:
        ab = 0

    V = len(ham)
    p = (ab + 1) / (a + V)
    return p

def sbprob(w, w1):
    try:
        a = spam[w]
    except KeyError:
        a = 0

    try:
        ab = spam[w, w1]
    except KeyError:
        ab = 0

    V = len(spam)
    p = (ab + 1) / (a + V)
    return p

A = 1
for bigram in sentence['bigram'][1]:
    A = A * hbprob(bigram[0], bigram[1])

print(A)

B=1
for bigram in sentence['bigram'][1]:
    B = B * sbprob(bigram[0], bigram[1])

print(B)

if A < B:
     print("This (Sorry, ..use your brain dear) message is  spam:",B)
else:
     print("This (Sorry, ..use your brain dear) message is  ham:",A)
#
# if A < B:
#      print("This (SIX chances to win CASH. message is  spam:",B)
# else:
#      print("This (SIX chances to win CASH.) message is  ham:",A)
#
# print(sms.shape)
# print (sms.head())
# print(sentence.head())