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

How I use lexicon for prediction:
- in each review, add a score of 1 for each positive word, and minus a score of 1 for each negative word, ignore those words that the lexicon doesn't have
- if the total score of a review is above a certain threshold (default is 0), we classify the review as positive, otherwise negative
- **Improvement**:

Negatives of this model: