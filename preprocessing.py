# ============ Text Pre-processhing Steps ============
# - Removing punctuations like . , ! $( ) * % @
# - Removing URLs
# - Removing Stop words
# - Lower casing
# - Tokenization
# - Stemming
# - Lemmatization

import string
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


stopwords = nltk.corpus.stopwords.words('english')
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

def remove_punctuation(text):
  punctuationfree="".join([i for i in text if i not in string.punctuation])
  return punctuationfree

def make_lowercase(text):
  return text.lower()

def tokenization(text):
  tokens = re.split('W+',text)
  return tokens[0]

def remove_whitespace(text: str):
  return text.strip()

def remove_stopwords(text):
  words = text.split()
  output = [i for i in words if i.strip() not in stopwords]
  return output

def stemming(text):
  stem_text = [porter_stemmer.stem(word) for word in text]
  return stem_text

def lemmatizer(text):
  lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
  return lemm_text


def preprocessing(text: str):
  removed_punc_text = remove_punctuation(text)
  lowercase_text = make_lowercase(removed_punc_text)
  tokenized_text = tokenization(lowercase_text)
  whitespace_removed_text = remove_whitespace(tokenized_text)
  removed_stopwords_text = remove_stopwords(whitespace_removed_text)
  # stemmed_text = stemming(removed_stopwords_text)
  lemmatized_text = lemmatizer(removed_stopwords_text)

  final_text = lemmatized_text

  return final_text