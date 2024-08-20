# libraries:
import pandas as pd
import warnings
from nltk.corpus import stopwords
from textblob import Word
import matplotlib.pyplot as plt
import seaborn as snn
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

##############
# READ DATASET
##############
df = pd.read_excel("nlp/datasets/amazon.xlsx")

df.columns = [col.upper() for col in df.columns]

####################
# DATA PREPROCESSİNG
####################

df.head()

# lower
def make_lower(dataframe, column):
    dataframe[column] = dataframe[column].str.lower()
    return dataframe


str_colums = [col for col in df.columns if df[col].dtype == "object"]

for i in str_colums:
    make_lower(df, i)

# remove punctuation marks:
for col in str_colums:
    df[col] = df[col].replace('[^\w\s]', "" , regex=True)

# extract numerical expressions:
for col in str_colums:
    df[col] = df[col].replace('\d', "" , regex=True)

# stop words:
sw = stopwords.words('english')
df['REVIEW'] = df['REVIEW'].apply(lambda x: ' '.join(word for word in str(x).split() if word not in sw))

#  Remove words with less than 1000 occurrences from the data:
temp_df = pd.Series(' '.join(df['REVIEW']).split()).value_counts()[-1000:]
df['REVIEW'] = df['REVIEW'].apply(lambda x: ' '.join(word for word in x.split() if word not in temp_df))

# Apply the lemmatization process:
df['REVIEW'] = df['REVIEW'].apply(lambda x: ' '.join(Word(w).lemmatize() for w in str(x).split()))

df['REVIEW'] = df['REVIEW'].apply(lambda x: ' '.join(Word(w).lemmatize() for w in str(x).split()))
df['REVIEW'].head(10)


####################
# TEXT VISUALİZATİON
####################

######################
# BARPLOT VISUALİZATİON
#######################

# frequency:
text = df['REVIEW'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
text.columns = ["WORDS", "TF"]

text[text["TF"] > 500].plot.barh(x="WORDS", y="TF")
plt.show()

# wordcloud:
text = " ".join(i for i in df['REVIEW'])
wordcloud = WordCloud().generate(text)

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

####################
# SENTİMENT ANALYSIS
####################

sia = SentimentIntensityAnalyzer()

sia.polarity_scores("I hate this machine")
sia.polarity_scores("This machine is very helpful for my job")

df['REVIEW'][:10].apply(lambda x: sia.polarity_scores(x)['compound'])
df['REVIEW'][:10].apply(lambda x: "pos" if sia.polarity_scores(x)['compound'] > 0 else "neg")

df['NEW_SENTİMENT_REVIEW'] = df['REVIEW'].apply(lambda x: "pos" if sia.polarity_scores(x)['compound']>0 else "neg")

df['NEW_SENTİMENT_REVIEW'].value_counts()

################################
# PREPARİNG FOR MACHİNE LEARNİNG
################################

X = df['REVIEW']
y = df['NEW_SENTİMENT_REVIEW']

# label encoder:
y = y.apply(lambda x: 1 if x == 'pos' else 0)

# train-test:
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2)

# Tfidf:
vectorizer = TfidfVectorizer().fit(X_train)
X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#################################
# MODELLİNG (LOGİSTİC REGRESSİON)
#################################

# model:
log_model = LogisticRegression().fit(X_train_tfidf, y_train)
y_pred = log_model.predict(X_test_tfidf)

# report:
print(classification_report(y_test, y_pred))

# cross val score:
cross_val_score(log_model, X_test_tfidf, y_test).mean()
# 0.85

# random review:
random_rewiew = pd.Series(df['REVIEW'].sample(1).values)
new_random = CountVectorizer().fit(X_train).transform(random_rewiew)

log_model.predict(new_random)

############################
# MODELLİNG (RANDOM FORREST)
############################

rf_model = RandomForestClassifier().fit(X_train_tfidf, y_train)

cross_val_score(rf_model, X_test_tfidf, y_test, cv=5, scoring="f1").mean()
# 0.93
