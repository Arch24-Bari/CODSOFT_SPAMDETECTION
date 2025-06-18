# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string
import nltk
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('spam.csv', encoding='latin-1')
df
# reading the CSV with a different encoding that should resolve this, else unicode error thrown

print(f'rows and columns are resp. {df.shape[0]} and {df.shape[1]}')
print('\n')
df.info()

print(df[df['Unnamed: 2'].notna()])

# unnamed columns don't contain a substantial data
# so we would drop the columns

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

print('Now,')
print(f'rows and columns are resp. {df.shape[0]} and {df.shape[1]}')
print('\n')
df.info()

df.rename(columns = {'v1':'target', 'v2':'msgs'}, inplace = True)
df

df.drop_duplicates(inplace=True)
df.duplicated().sum()
print(f'rows and columns are resp. {df.shape[0]} and {df.shape[1]}')

plt.title('Denomination of Ham and Spam msgs')
denom = df['target'].value_counts()
plt.pie(denom, labels = denom.index, autopct='%1.1f%%')
plt.show()
# some cells might be an extension of our data analysis, in the text pre processing section

"""**TEXT PREPROCESSING**"""

import nltk
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

def modf_text(text):

  # tokenise the text = breaking down into words
  text = nltk.word_tokenize(text)
  # lower case conversion of words with removal of special characters ( not alphanumeric )
  text = [word.lower() for word in text if word.isalpha()]
  # removing stopwords
  text = [word for word in text if word not in stopwords.words('english')]
  # remove punctuation marks
  text = [word for word in text if word not in string.punctuation]
  # stemming
  ps = nltk.PorterStemmer()
  text = [ps.stem(word) for word in text]
  # return the modified text
  return ' '.join(text)

# testing a sample text
modf_text('Hi @Sma YUIIO, I am not @taKIN your gIft*!!!')

print('Following are stopwords, refer for understanding')
# uncomment below to read
# stopwords.words('english')[]

# applying that to the msgs column
df['msgs'] = df['msgs'].apply(modf_text)
df

# Label encoding of 'target' columns

from sklearn.preprocessing import LabelEncoder
le  = LabelEncoder()
df['target'] = le.fit_transform(df['target'])
df.head()

# ham = 0, spam = 1

wc = WordCloud(height =250, width = 250, min_font_size=15, background_color='white')
spam_wc = wc.generate(df[df['target'] == 1]['msgs'].str.cat(sep=' '))
ham_wc = wc.generate(df[df['target'] == 0]['msgs'].str.cat(sep=' '))


plt.title('Word Cloud for Spam msgs')
plt.imshow(spam_wc)


plt.show()

plt.title('Word Cloud for Ham msgs')
plt.imshow(ham_wc)

sp_w = df[df['target'] == 1]['msgs'].tolist()
h_w = df[df['target'] == 0]['msgs'].tolist()

spam_corpus = [word for msg in sp_w for word in msg.split()]
ham_corpus = [word for msg in h_w for word in msg.split()]
Counter(spam_corpus).most_common(20)

Counter(ham_corpus).most_common(20)

s_df = pd.DataFrame(Counter(spam_corpus).most_common(20), columns=['words', 'count'])
s_df
#returns word count of highest occuring words

h_df = pd.DataFrame(Counter(ham_corpus).most_common(20), columns  = ['words' , 'count'])
h_df

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Count of most occuring words in spam messages')
plt.ylabel('Frequency of usage')
sns.barplot(data = s_df, x= 'words', y = 'count')
plt.xticks(rotation = 90)
plt.subplot(1, 3, 3)
plt.title('Count of most occuring words in ham messages')
plt.ylabel('Frequency of usage')
sns.barplot(data = h_df, x='words', y = 'count', color='orange')
plt.xticks(rotation = 90)
plt.show()

"""**SPLITTING DATA**

---count vectoriser---
"""

# First we would experiment with Count vectorizing method and then tfidf one
cv = CountVectorizer()
X = cv.fit_transform(df['msgs']).toarray()
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(X_train, y_train)
mnb.fit(X_train, y_train)
bnb.fit(X_train, y_train)
y_pred_1 = gnb.predict(X_test)
y_pred_2 = mnb.predict(X_test)
y_pred_3 = bnb.predict(X_test)

print(f'GaussianNB accuracy: {accuracy_score(y_test, y_pred_1)}')
print(f'GaussianNB precision:{precision_score(y_test, y_pred_1)}')
print(' \n')
print(f'MultinomialNB accuracy: {accuracy_score(y_test, y_pred_2)}')
print(f'MultinomialNB precision:{precision_score(y_test, y_pred_2)}')

print(' \n')
print(f'BernoulliNB accuracy: {accuracy_score(y_test, y_pred_3)}')
print(f'BernoulliNB precision:{precision_score(y_test, y_pred_3)}')

# Commented out IPython magic to ensure Python compatibility.
# %pip install xgboost

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

lr = LogisticRegression(solver = 'liblinear', penalty= 'l1')
dt = DecisionTreeClassifier(max_depth = 5)
rf = RandomForestClassifier(n_estimators=50, random_state = 2)
gbdt = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, random_state=2)
xgb = XGBClassifier(n_estimators=50, learning_rate=0.7, random_state=2)
svc = SVC(kernel='sigmoid', gamma=1.0)
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
gbdt.fit(X_train, y_train)
xgb.fit(X_train, y_train)

for clf in (lr, dt, rf, gbdt, xgb):
  y_pred = clf.predict(X_test)
  print(f'{clf.__class__.__name__} accuracy: {accuracy_score(y_test, y_pred)}')
  print(f'{clf.__class__.__name__} precision: {precision_score(y_test, y_pred)}')
  print(' \n')

li_a = [accuracy_score(y_test, clf.predict(X_test)) for clf in (lr, dt, rf, gbdt, xgb)]
li_p = [precision_score(y_test, clf.predict(X_test)) for clf in (lr, dt, rf, gbdt, xgb)]
d_p = pd.DataFrame(li_p, index = ('lr', 'dt', 'rf', 'gbdt', 'xgb'), columns=['precision'])
d_a = pd.DataFrame(li_a, index = ('lr', 'dt', 'rf', 'gbdt', 'xgb'), columns=['accuracy'])
d = pd.concat([d_a, d_p], axis=1).sort_values('accuracy', ascending=False)
d

"""Performance of model determined by precision of the model, there shudn't b False positives as per the case study

Thus, Random Forest Model is the best one if CountVectorizer() is used

Below Is a graph for more detailed understanding
"""

plt.figure(figsize=(10, 8))
sns.barplot(x=d.index, y=d['precision'], hue = d['accuracy'], legend= 'brief', )
plt.tight_layout()
plt.title('Precision and Accuracy, accuracy is the hue')
plt.show()

"""---TF-IDF vectorizer---"""

tf = TfidfVectorizer()
X =tf.fit_transform(df['msgs']).toarray()
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)
X

gnb_1 = GaussianNB()
mnb_1 = MultinomialNB()
bnb_1 = BernoulliNB()

gnb_1.fit(X_train, y_train)
mnb_1.fit(X_train, y_train)
bnb_1.fit(X_train, y_train)
y_pred_1_ = gnb.predict(X_test)
y_pred_2_ = mnb.predict(X_test)
y_pred_3_ = bnb.predict(X_test)

print(f'GaussianNB accuracy: {accuracy_score(y_test, y_pred_1_)}')
print(f'GaussianNB precision:{precision_score(y_test, y_pred_1_)}')
print(' \n')
print(f'MultinomialNB accuracy: {accuracy_score(y_test, y_pred_2_)}')
print(f'MultinomialNB precision:{precision_score(y_test, y_pred_2_)}')

print(' \n')
print(f'BernoulliNB accuracy: {accuracy_score(y_test, y_pred_3_)}')
print(f'BernoulliNB precision:{precision_score(y_test, y_pred_3_)}')
#Multinomial NB may be the most preferred model in this approach

lr = LogisticRegression(solver = 'liblinear', penalty= 'l1')
dt = DecisionTreeClassifier(max_depth = 5)
rf_1 = RandomForestClassifier(n_estimators=50, random_state = 2)
gbdt = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, random_state=2)
xgb = XGBClassifier(n_estimators=50, learning_rate=0.7, random_state=2)
svc = SVC(kernel='sigmoid', gamma=1.0)
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf_1.fit(X_train, y_train)
gbdt.fit(X_train, y_train)
xgb.fit(X_train, y_train)

li_a_1 = [accuracy_score(y_test, clf.predict(X_test)) for clf in (lr, dt, rf, gbdt, xgb)]
li_p_1 = [precision_score(y_test, clf.predict(X_test)) for clf in (lr, dt, rf, gbdt, xgb)]
d_p_1 = pd.DataFrame(li_p_1, index = ('lr', 'dt', 'rf', 'gbdt', 'xgb'), columns=['precision'])
d_a_1 = pd.DataFrame(li_a_1, index = ('lr', 'dt', 'rf', 'gbdt', 'xgb'), columns=['accuracy'])
d_1 = pd.concat([d_a_1, d_p_1], axis=1).sort_values('accuracy', ascending=False)
d_1

plt.figure(figsize=(10, 8))
sns.barplot(x=d_1.index, y=d_1['precision'], hue = d_1['accuracy'], legend= 'brief', )
plt.tight_layout()
plt.title('Precision vs Accuracy')
plt.show()
#Rf model may also be preferred

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

"""IMPROVEMENT THROUGH VOTING CLASSIFIER APPROACH"""

from sklearn.ensemble import VotingClassifier
# Using Logistic Regression, Multinomial Naive Bayes, and Random Forest based on previous results
estimators = [('bnb', bnb_1), ('mnb', mnb_1), ('rf', rf)]

voting_clf = VotingClassifier(estimators=estimators, voting='hard')

voting_clf.fit(X_train, y_train)

y_pred_voting = voting_clf.predict(X_test)

print(f'Voting Classifier accuracy: {accuracy_score(y_test, y_pred_voting)}')
print(f'Voting Classifier precision: {precision_score(y_test, y_pred_voting)}')

from sklearn.ensemble import VotingClassifier

estimators_soft = [('lr', lr) ,('mnb', mnb_1), ('bnb', bnb_1), ('xgb', xgb), ('rf', rf),('rf1', rf_1)]


voting_clf_soft = VotingClassifier(estimators=estimators_soft, voting='soft')

voting_clf_soft.fit(X_train, y_train)

y_pred_voting_soft = voting_clf_soft.predict(X_test)

# Evaluate the voting classifier
print(f'Voting Classifier (Soft Voting) accuracy: {accuracy_score(y_test, y_pred_voting_soft)}')
print(f'Voting Classifier (Soft Voting) precision: {precision_score(y_test, y_pred_voting_soft)}')
# combo of the models along with soft voting approach, resulting in model to be the most effective discrimanatory model from the pool of models provided

"""In the case of imbalanced dataset we are more concerned with the precision score obtained through models

IMPROVEMENT THROUGH STACK CLASSIFIER APPROACH (trial, may not improve though)
"""

estimators_st = [('mnb', mnb_1), ('bnb', bnb_1),('rf1', rf_1)]
final_estim = RandomForestClassifier()
stack_clf = StackingClassifier(estimators = estimators_st, final_estimator = final_estim)
stack_clf.fit(X_train, y_train)
y_pred_stack = stack_clf.predict(X_test)

# Evaluate the voting classifier
print(f'Stacking Classifier (Soft Voting) accuracy: {accuracy_score(y_test, y_pred_stack)}')
print(f'Stacking Classifier (Soft Voting) precision: {precision_score(y_test, y_pred_stack)}')

# no better result obtained from this

"""CONCLUSION

Soft Voting Classifier appear to be the most robust model for discrimination b/w spam and ham messages for the dataset

This achieves perfect precision on the test while maintaining high accuracy

TEST RUNS ON THE MODEL
"""

voting_clf_soft.predict(tf.transform(['free wkli comp win fa cup final tkt may ']))
# array([1]) = spam message

voting_clf_soft.predict(tf.transform(['new servic price get']))

voting_clf_soft.predict(tf.transform(['one lor love ok']))

