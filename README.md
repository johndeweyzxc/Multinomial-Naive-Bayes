# About

This project implements the **Multinomial Naive Bayes Machine Learning Algorithm from scratch**. The goal is to delve deep into the mathematical foundations of the Naive Bayes algorithm. Using data from Reddit to train the model, the algorithm predicts the subreddit (class) based on the text and title of submission posts. The datasets folder contains the data for 8 target subreddits from January 2024 to September 2024. Additionally, this project also implements the algorithm from scikit learn machine learning library. The dataset came from Pushshift which can be downloaded here: https://academictorrents.com/details/ac88546145ca3227e2b90e51ab477c4527dd8b90

## Multinomial Naive Bayes Classifier

A Multinomial Naive Bayes Classifier is a type of Naive Bayes algorithm used primarily for text classification tasks, such as spam detection, sentiment analysis, and document categorization. It is well-suited for data where the features represent counts or frequencies, such as the number of times a word appears in a document. The algorithm is based on Bayes' Theorem, assuming that the features are conditionally independent given the class.

$$\large \^C = \underset{C}{\operatorname{argmax}} [\sum_{i=1}^{n}log(P(x_i|C_i))]$$

Where $P(x_i|C_i)$ is the probability of a term $x_i$ given class $C_i$ and $n$ is the total number of terms in a document. Taking the logarithm of probabilities (log-likelihood) helps in numerical stability and simplifies calculations. The classifier predicts the class that has the highest log-probability score given the set of features.
