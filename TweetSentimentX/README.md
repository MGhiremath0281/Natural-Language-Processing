# TweetSentimentX

**TweetSentimentX** is a sentiment analysis model that classifies tweets into four sentiment categories: **Positive**, **Negative**, **Neutral**, and **Irrelevant**. The model is built using **Logistic Regression** and **TF-IDF** for text feature extraction.

## Model Overview

The model predicts the sentiment of tweets by processing the text and classifying it into one of the following categories:

- **Positive**: The tweet expresses a favorable sentiment.
- **Negative**: The tweet expresses an unfavorable sentiment.
- **Neutral**: The tweet is neither positive nor negative.
- **Irrelevant**: The tweet does not contain sentiment-relevant content.

### Approach

1. **Data Collection & Preprocessing**: 
   - The dataset contains tweets labeled with sentiments.
   - Preprocessing steps such as tokenization, stop-word removal, and lemmatization were applied to clean and standardize the text.

2. **Feature Extraction (TF-IDF)**:
   - **TF-IDF** (Term Frequency-Inverse Document Frequency) is used to transform the tweet text into numerical features, representing the importance of each word in the text relative to the corpus.

3. **Model Training (Logistic Regression)**:
   - A **Logistic Regression** model was trained on the preprocessed data using the TF-IDF features. Logistic Regression was chosen due to its simplicity and effectiveness for text classification tasks.
   - The model was evaluated with an accuracy of **77%**, indicating reliable sentiment predictions.

4. **Prediction**:
   - The trained model predicts the sentiment of new, unseen tweets by transforming the input text into features and making a classification based on learned patterns.

## Technologies Used

- **Flask**: Lightweight web framework for creating the user interface.
- **scikit-learn**: Library for machine learning, used for model training and evaluation.
- **TF-IDF**: Text vectorization technique used for feature extraction.
- **Logistic Regression**: Classification algorithm used for sentiment prediction.

## Performance

- **Accuracy**: 77%
- The model performs well in classifying tweets, with a balanced distribution of positive, negative, neutral, and irrelevant sentiments.

## Conclusion

**TweetSentimentX** provides an efficient and easy-to-use platform for analyzing tweet sentiment. By leveraging TF-IDF for feature extraction and Logistic Regression for classification, the model is capable of providing reliable sentiment predictions for real-time tweet analysis.

