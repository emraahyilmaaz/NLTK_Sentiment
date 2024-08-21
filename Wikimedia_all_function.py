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


def wiki_preprocess(text, Barplot=False, Wordcloud=False):
    """
    pre-processing on texts

    :param text: columns that contains text
    :param Barplot: visualization text data as a bar plot
    :param Wordcloud: visualization text data as a picture
    :return: text
    """

    # make lower sentences:
    text = text.str.lower()

    # remove punctuation:
    text = text.str.replace('[^\w\s]', '', regex=True)
    text = text.str.replace(r'\n\n', ' ', regex=True)

    # remove numeric
    text = text.str.replace('\d', "" , regex=True)

    # stopwords:
    sw = stopwords.words("english")
    text = text.apply(lambda x: " ".join(word for word in x.split() if word not in sw))

    # only mentioned once in the text:
    word_count = pd.Series(" ".join(text).split()).value_counts()
    single_occurrences = word_count[word_count == 1].index

    # remove single_occurrences:
    text = text.apply(lambda x: " ".join(word for word in str(x).split() if word not in single_occurrences))

    # lemmatization:
    text = text.apply(lambda x: " ".join(Word(w).lemmatize() for w in str(x).split()))

    if Barplot:
        # frequency:
        tf = text.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        tf.columns = ["words", "tf"]

        # bar plot:
        tf.sort_values(ascending=False, by="tf").head(15).plot.barh(x="words", y="tf")
        plt.show()

    if Wordcloud:
        # join words:
        text = " ".join(i for i in text)

        # mask:
        mask = np.array(Image.open('/Users/emrahyilmaz/streamlet'
                                   '/NLP/nlp/outline-drawing-world-map-black-260nw-517822060.jpg'))

        cloud = WordCloud(scale=3,
                          max_words=150,
                          colormap='RdYlGn',
                          mask=mask,
                          collocations=True).generate_from_text(text)
        plt.figure(figsize=(10, 8))
        plt.imshow(cloud)
        plt.axis('off')
        plt.title('PİCTURE OF WIKIPEDİA TEXT')
        plt.show()

    return text