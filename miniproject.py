import jsonlines
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import regex as re
from collections import Counter

#nltk.download('punkt_tab')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')

data_list = []

with jsonlines.open("data/train.jsonl", mode='r') as reader:
    for obj in reader:
        obj.pop('model')
        obj.pop('genre')
        obj.pop('id')
        data_list.append(obj)

df = pd.DataFrame(data_list)

tokens_all = [word_tokenize(sentence) for sentence in df.text]

# Removing all tokens consisting only of non-alphanumeric characters, that aren't inbetween words e.g. "battle-field",
# so any tokens such as: "''", "--", ".", "," etc. are removed, but "battle-field" is not removed. 
for sentence in tokens_all:
    while True:
        for token in sentence:
            if re.search(r"\B[^a-zA-Z0-9]+\B", token) != None:
                sentence.remove(token) # For each token in a sentence, if the token consists only of 1 or more non-alphanumeric characters, remove it from the sentence.

        outs = [] # Creates an empty list. The list is reset to empty every iteration of the while-loop. 
        for token in sentence:
            if re.search(r"\B[^a-zA-Z0-9]+\B", token) != None:
                outs.append(token) # This for-loop makes an additional check for any non-alphanumeric tokens, and appends then to the empty list 'outs'.

        if outs: # This checks whether the list 'outs' is empty or not. If it's not empty, then 'continue' is activated and the while-loop continues to iterate,
            continue # if the list is empty, then 'break' is activated, and the while-loop terminates, and the outer-most for-loop continues to the next 'sentence' in 'tokens_all'.
        else:
            break

word_list = [[word.lower() for word in sentence] for sentence in tokens_all]

words = []
for sentence in word_list:
    for word in sentence:
        words.append(word)

len(words)

vocab = sorted(set(words))

len(vocab)

freq = Counter(words)

stop_words = set(stopwords.words('english'))

len(stop_words)

filtered_tokens = [word for word in words if word not in stop_words]
# This keeps the structure of a list with all datapoints in the train set, where each datapoint
# is in the form of a list. So it's a list with 23707 lists/data points with tokens.
# Each token has been converted to small characters with word.lower(). 

filtered_token_list = [[token for token in sentence if token not in stop_words] for sentence in word_list]
# Stop-words have now been removed as well. 

filtered_token_strings = [" ".join(sentence) for sentence in filtered_token_list]

print(df)

df['text'] = filtered_token_strings

print(df)

filtered_vocab = sorted(set(filtered_tokens))

tags = pos_tag(filtered_tokens, tagset='universal')


tag_list = [pos_tag(sentence, tagset='universal') for sentence in filtered_token_list]


sentence_tags = [[token[-1] for token in sentence] for sentence in tag_list]

print(sentence_tags[:2])

unique_tags = set([tag[-1] for tag in tags])

len(unique_tags)

tags_freq = [dict(Counter(sentence)) for sentence in sentence_tags]

possible_tags = list(unique_tags)

print(possible_tags)

missing_tags = [{tag: 0 for tag in possible_tags if tag not in sentence} for sentence in tags_freq]

len(missing_tags)

print(missing_tags[0])

print(tags_freq[0])

all_tags = []

for i in range(len(missing_tags)):
    tags = tags_freq[i]
    missing = missing_tags[i]
    tags.update(missing)
    all_tags.append(tags)

print(all_tags[0])

# Sorting the dict, 'example_tags', by making a dict comprehension and using sorted() with lambda
# inserted in the 'key'-argument of the sorted() function.
tags_sorted = [{k: v for k, v in sorted(sentence.items(), key=lambda item: item[0])} for sentence in all_tags]
# key=lambda item: item[0] is exactly the same as writing:
# key = def lambda(item):
#           return item[0]
# lambda is a throwaway function, just like _ is a throwaway variable.

print(tags_sorted[:2])
len(tags_sorted)

tag_values = [list(sentence.values()) for sentence in tags_sorted]
len(tag_values)

print(tag_values[:2])

print(df)

df['POS_vector'] = tag_values

print(df)


dev_data = []

with jsonlines.open("data/dev.jsonl", mode='r') as doc:
    for obj in doc:
        obj.pop('model')
        obj.pop('genre')
        obj.pop('id')
        dev_data.append(obj)

dev_df = pd.DataFrame(dev_data)

len(dev_df)

print(dev_df)

dev_tokens = [word_tokenize(sentence) for sentence in dev_df.text]

# Removing all tokens consisting only of non-alphanumeric characters, that aren't inbetween words e.g. "battle-field",
# so any tokens such as: "''", "--", ".", "," etc. are removed, but "battle-field" is not removed. 
for sentence in dev_tokens:
    while True:
        for token in sentence:
            if re.search(r"\B[^a-zA-Z0-9]+\B", token) != None:
                sentence.remove(token) # For each token in a sentence, if the token consists only of 1 or more non-alphanumeric characters, remove it from the sentence.

        outs = [] # Creates an empty list. The list is reset to empty every iteration of the while-loop. 
        for token in sentence:
            if re.search(r"\B[^a-zA-Z0-9]+\B", token) != None:
                outs.append(token) # This for-loop makes an additional check for any non-alphanumeric tokens, and appends then to the empty list 'outs'.

        if outs: # This checks whether the list 'outs' is empty or not. If it's not empty, then 'continue' is activated and the while-loop continues to iterate,
            continue # if the list is empty, then 'break' is activated, and the while-loop terminates, and the outer-most for-loop continues to the next 'sentence' in 'tokens_all'.
        else:
            break




tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding="utf-8", ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.text).toarray()
labels = df.label

features.shape # Now each of the 23707 sequences of text is represented by 195904 features, 
               # representing the tf-idf score for different unigrams and bigrams.

model = LogisticRegression(random_state=42)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))