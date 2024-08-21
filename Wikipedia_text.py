##############################
# WİKİPEDİA TEXT VISUALIZATION
##############################

# libraries:
import pandas as pd
import warnings
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

####################
# DATA PREPROCESSİNG
####################

df_ = pd.read_csv('/Users/emrahyilmaz/Downloads/wiki-221126-161428/wiki_data.csv', index_col=0)
df = df_.copy()
df.columns = ['TEXT']

# cleaning:
def clean_text(text):
    text = text.str.lower() # make lower sentences
    # remove punctuation:
    text = text.str.replace('[^\w\s]', '', regex=True)
    text = text.str.replace(r'\n\n', ' ', regex=True)

    text = text.str.replace('\d', "" , regex=True) # remove numeric

    return text


df['TEXT'] = clean_text(df['TEXT'])

# stop words:
def remove_stopwords(text_column):
    sw = stopwords.words("english")
    text_column = text_column.apply(lambda x: " ".join(word for word in x.split() if word not in sw))
    return text_column


df['TEXT'] = remove_stopwords(df['TEXT'])

# only mentioned once in the text:
word_count = pd.Series(" ".join(df['TEXT']).split()).value_counts()
single_occurrences = word_count[word_count == 1].index

# remove single_occurrences:
df['TEXT'] = df['TEXT'].apply(lambda x: " ".join(word for word in str(x).split() if word not in single_occurrences))

# tokenization:
df['TEXT'].apply(lambda x: TextBlob(x).words)

#  Lemmatization:
df['TEXT'] = df['TEXT'].apply(lambda x: " ".join(Word(w).lemmatize() for w in str(x).split()))


####################
# DATA VISUALIZATION
####################

# frequency:
tf = df['TEXT'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.sort_values(ascending=False, by="tf")

# bar plot:
tf.sort_values(ascending=False, by="tf").head(15).plot.barh(x="words", y="tf")
plt.show()

# join words:
text = " ".join(i for i in df['TEXT'])

# wordcloud:
mask = np.array(Image.open('/Users/emrahyilmaz/streamlet/NLP/nlp/outline-drawing-world-map-black-260nw-517822060.jpg'))

def generate_better_wordcloud(data, title, mask=None):
    cloud = WordCloud(scale=3,
                      max_words=150,
                      colormap='RdYlGn',
                      mask=mask,
                      collocations=True).generate_from_text(data)
    plt.figure(figsize=(10,8))
    plt.imshow(cloud)
    plt.axis('off')
    plt.title(title)
    plt.show()


generate_better_wordcloud(text, "wikipedia", mask)
