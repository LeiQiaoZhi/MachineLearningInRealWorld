# Machine Learning In Real World
*files of the exercises of my first year machine learning course at the University of Cambridge*

## Folders and What They Contain
- `data`: training, validation and testing data for two tasks:
	1. Sentiment Classification (NLP)
	2. Hidden States Prediction (Markov Model)
- `exercises`: code that I wrote for the exercises(we call it 'ticks')  
- `utils`: code for utility, such as graph sketching and printing matrices

## What I Did and Learnt
### Tick 1
**Task**: Sentiment Prediction of movie reviews using lexicon
- classify into 2 classes: `Positive` and `Negative`
- a **lexicon** is a dictionary of words and sentiment
*e.g. {'fantastic':1,'bad':-1}*

#### How I used lexicon for prediction:
- in each review, add a score of 1 for each positive word, and minus a score of 1 for each negative word, ignore those words that the lexicon doesn't have
- if the total score of a review is above a certain threshold (default is 0), we classify the review as positive, otherwise negative
- **Improvement**:

#### Negatives of this model:
---
### Tick 2
Task: **Naive Bayes Classification** for movie reviews sentiment classification

#### Naive Bayes:
- **Bayes Law**: $P(C|w_0...w_n)=\frac{P(w_0...w_n|C)P(C)}{P(w_0...w_n)}$
- **Naive Assumption**: every word occurence is independent from each other, thus $P(w_0...w_n|C)=P(w_0|C)...P(w_n|C)$

How to classify:
- compare $P(C|w_0...w_n)$ for every class $C$, in this task the classes are *Positive* and *Negative*, the class with the highest probability is the model's prediction
- since $P(w_0...w_n)$ is constant for every class, we just need to compare $P(C)P(w_0|C)...P(w_n|C)$

How to estimate $P(C)$:
- $P(C) = \frac{\text{number of training data labelled C}}{\text{number of all training data}}$

How to estimate $P(w_0|C)$:
- $$P(w_0|C)=\frac{\text{number of $w_0$'s occurences in training data labelled C}}{\text{number of all features' occurenes in training data labelled C}}$$

> In practice, I used $\log$ so the model compares $\log(P(C)P(w_0|C)...P(w_n|C))$$=\log(P(C))+\log(P(w_0|C))+...+\log(P(w_n|C))$