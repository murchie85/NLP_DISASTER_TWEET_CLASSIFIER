# NLP DISASTER TWEET RECOGNITION 


```python

import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing, metrics
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df.head()
```


## OUTPUT 


```

id	keyword	location	text	target
0	1	NaN	NaN	Our Deeds are the Reason of this #earthquake M...	1
1	4	NaN	NaN	Forest fire near La Ronge Sask. Canada	1
2	5	NaN	NaN	All residents asked to 'shelter in place' are ...	1
3	6	NaN	NaN	13,000 people receive #wildfires evacuation or...	1
4	7	NaN	NaN	Just got sent this photo from Ruby #Alaska as ...	1

```



```python
train_df[train_df['target'] == 0].head()
```

## OUTPUT 

```

id	keyword	location	text	target
15	23	NaN	NaN	What's up man?	0
16	24	NaN	NaN	I love fruits	0
17	25	NaN	NaN	Summer is lovely	0
18	26	NaN	NaN	My car is so fast	0
19	28	NaN	NaN	What a goooooooaaaaaal!!!!!!	0
```

```python

print('this is an example of a disaster tweet')
print(train_df[train_df['target'] ==1]['text'].values[1])

print('this is an example of a non-disaster tweet')
print(train_df[train_df['target'] ==0]['text'].values[1])

```

## OUTPUT 

```
this is an example of a disaster tweet
Forest fire near La Ronge Sask. Canada
this is an example of a non-disaster tweet
I love fruits

```

# Building vectors

The theory behind the model we'll build in this notebook is pretty simple: the words contained in each tweet are a good indicator of whether they're about a real disaster or not (this is not entirely correct, but it's a great place to start).

We'll use scikit-learn's CountVectorizer to count the words in each tweet and turn them into data our machine learning model can process.

Note: a vector is, in this context, a set of numbers that a machine learning model can work with. We'll look at one in just a second.


```python
count_vectorizer = feature_extraction.text.CountVectorizer()

## let's get counts for the first 5 tweets in the data
example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])

## we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)
print(example_train_vectors[0].todense().shape)
```

## output 

```
(1, 54)
```

```python
print(example_train_vectors[0].todense())
```

## output 

```
[[0 0 0 1 1 1 0 0 0 0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0
  0 0 0 1 0 0 0 0 0 0 0 0 0 1 1 0 1 0]]
```

The above tells us that:

There are **54 unique words (or "tokens") in the first five tweets.**  
The first tweet contains only some of those unique tokens - all of the non-zero counts above are the tokens that DO exist in the first tweet.
Now let's create vectors for all of our tweets.

```python
inputData["text"][0:5]
train_vectors = count_vectorizer.fit_transform(train_df["text"])

## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
# that the tokens in the train vectors are the only ones mapped to the test vectors - 
# i.e. that the train and test vectors use the same set of tokens.
test_vectors = count_vectorizer.transform(test_df["text"])
```


# Our model
As we mentioned above, we think the words contained in each tweet are a good indicator of whether they're about a real disaster or not. The presence of particular word (or set of words) in a tweet might link directly to whether or not that tweet is real.  

What we're assuming here is a linear connection. So let's build a linear model and see!

## BUILDING A LINEAR MODEL 

```python
## Our vectors are really big, so we want to push our model's weights
## toward 0 without completely discounting different words - ridge regression 
## is a good way to do this.
clf = linear_model.RidgeClassifier()
```


Let's test our model and see how well it does on the training data. For this we'll use `cross-validation` - where we train on a portion of the known data, then validate it with the rest. If we do this several times (with different portions) we can get a good idea for how a particular model or method performs.  

The metric for this competition is F1, so let's use that here.


```python
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")
scores
```

## output 

```
array([0.60355649, 0.57580105, 0.64516129])
```


The above scores aren't terrible! It looks like our assumption will score roughly 0.65 on the leaderboard. There are lots of ways to potentially improve on this (TFIDF, LSA, LSTM / RNNs, the list is long!) - give any of them a shot!  

In the meantime, let's do predictions on our training set and build a submission for the competition.

```python
clf.fit(train_vectors, train_df["target"])
```

## output 

```
RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
        max_iter=None, normalize=False, random_state=None, solver='auto',
        tol=0.001)
```

## FIT on sample submission data

```python
sample_submission = pd.read_csv("sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.head()
```

## output 
```
	id	target
0	0	0
1	2	1
2	3	1
3	9	0
4	11	1
```

## SAVE SUBMISSION

```python
sample_submission.to_csv("submission.csv", index=False)
```





