# About

- This project implements the **Multinomial Naive Bayes (MNB) Machine Learning Algorithm from scratch** to gain a deeper understanding of its mathematical foundations.
- Using data from Reddit, the objective is to predict the subreddit (class) based on submission post title and text content.
- The dataset, covering 8 subreddits of interest from January 2024 to September 2024, is available in the datasets folder.
- Additionally, this project implements the scikit-learn version of the MNB algorithm for comparison of results.
- The dataset is sourced from Pushshift, available for download here: https://academictorrents.com/details/ac88546145ca3227e2b90e51ab477c4527dd8b90

## Multinomial Naive Bayes Classifier

A Multinomial Naive Bayes Classifier is a type of Naive Bayes algorithm used primarily for text classification tasks, such as spam detection, sentiment analysis, and document categorization. It is well-suited for data where the features represent counts or frequencies, such as the number of times a word appears in a document. The algorithm is based on Bayes' Theorem, assuming that the features are conditionally independent given the class.

<br>
The model is mathematically represented as:
<br>

$$\large \hat C = \underset{C}{\text{argmax}} \big[\small \sum_{i=1}^{n}log(P(x_i|C_i))\big]$$

Where:

- $\hat C$ is the predicted class
- $P(x_i|C_i)$ is the probability of feature $x_i$ (term) given class $C_i$.
- $n$ is the total number of terms in a document.

Taking the logarithm of probabilities (log-likelihood) helps in numerical stability and simplifies calculations. **The classifier predicts the class that has the highest log-probability score given the set of features.**

## Evaluation

To assess the performance of the Multinomial Naive Bayes classifier, we evaluated the model on a test set containing posts from 8 different subreddits. The evaluation metrics used are **precision, recall, F1-score, and accuracy**. We compare both the **scratch implementation** of the algorithm and the implementation from **scikit-learn**.

### Scratch Implementation

                         precision    recall  f1-score   support

       CryptoCurrencies       0.86      0.71      0.78       400
           DeepThoughts       0.65      0.77      0.71       400
              LawSchool       0.73      0.89      0.80       400
    PoliticalDiscussion       0.87      0.67      0.76       400
      Wallstreetbetsnew       0.74      0.80      0.77       400
          askphilosophy       0.75      0.82      0.78       400
        computerscience       0.90      0.73      0.81       400
            geopolitics       0.81      0.84      0.83       400

               accuracy                           0.78      3200
              macro avg       0.79      0.78      0.78      3200
           weighted avg       0.79      0.78      0.78      3200

### Scikit-Learn Implementation

                         precision    recall  f1-score   support

       CryptoCurrencies       0.78      0.80      0.79       400
           DeepThoughts       0.67      0.69      0.68       400
              LawSchool       0.75      0.88      0.81       400
    PoliticalDiscussion       0.85      0.70      0.77       400
      Wallstreetbetsnew       0.81      0.80      0.80       400
          askphilosophy       0.74      0.84      0.79       400
        computerscience       0.90      0.72      0.80       400
            geopolitics       0.83      0.87      0.85       400

               accuracy                           0.79      3200
              macro avg       0.79      0.79      0.79      3200
           weighted avg       0.79      0.79      0.79      3200

The overall accuracy of the scratch implementation is 78%, indicating a reasonably good classification performance across all subreddits.

### Confusion Matrix

Below are the confusion matrices for both the scratch and scikit-learn implementations. These matrices provide insight into which subreddits the model struggles with the most, by showing how many posts from one subreddit were misclassified into another.

<div align="center">
  <img src="/metrics/confusion-matrix/scratch/mnb_conf_matrix_scratch1.png" width=700 />
  <img src="/metrics/confusion-matrix/sklearn/mnb_conf_matrix_sklearn1.png" width=700 />
</div>
