import warnings
warnings.filterwarnings('ignore')

import emoji
import string
import nltk
# from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

import re

import pickle  

stop_words = set(stopwords.words('english'))

def strip_emoji(text):
    return emoji.replace_emoji(text,replace="")

def strip_all_entities(text):
    
    text= text.replace('\r', '').replace('\n','').lower()
    text= re.sub(r"(?:\@|https?|-\://)\S+",'',text)
    text= re.sub(r"[^\x00-\x7f]",r'',text)
    text= re.sub('[0-9]+','',text)
    
    stopchars =string.punctuation
    table=str.maketrans('','',stopchars)
    text=text.translate(table)
    
    text=[word for word in text.split() if word not in stop_words]
    text=' '.join(text)
    
    return text

def decontract(text):
    
    text=re.sub(r"cant\'t'" ,"can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    
    return text

def clean_hashtags(tweet):
    
    new_tweet=" ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet))
    
    return new_tweet2

def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

def remove_mult_spaces(text):
    return re.sub("\s\s+" , " ", text)

def stemmer(text):
    tokenized = nltk.word_tokenize(text)
    ps = PorterStemmer()
    return ' '.join([ps.stem(words) for words in tokenized])

def lemmatize(text):
    tokenized = nltk.word_tokenize(text)
    lm = WordNetLemmatizer()
    return ' '.join([lm.lemmatize(words) for words in tokenized])

def preprocess(text):
    text = strip_emoji(text)
    text = decontract(text)
    text = strip_all_entities(text)
    text = clean_hashtags(text)
    text = filter_chars(text)
    text = remove_mult_spaces(text)
    text = stemmer(text)
    text = lemmatize(text)
    return text

with open('cyberbullyapp/utils/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('cyberbullyapp/utils/model.pkl', 'rb') as f:
    model = pickle.load(f)



def sentiment_classification(input_value):

    out=preprocess(input_value)



    

    x_text=[out]

    x_text=vectorizer.transform(x_text)

    pred=model.predict(x_text)

    mapping_dict={
    1:'religion',
    2:'age',
    3:'ethnicity',
    4:'gender',
    5:'other_cyberbullying',
    6:'not_cyberbullying',
    }

    return mapping_dict[pred[0]]


    

