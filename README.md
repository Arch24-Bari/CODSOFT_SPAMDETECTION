# CODSOFT_SPAMDETECTION
THIS CONTAINS THE SPAM_DETECTION MACHINE LEARNING PROJECT
Following were the flow of computations performed under the project
* successfully loaded and preprocessed the spam dataset, handling missing values and duplicates.
* gained insights into the most frequent words in both spam and ham messages through word clouds and frequency counts.
* experimented with both CountVectorizer and TF-IDFVectorizer for text representation.
* trained and evaluated various classification models (Naive Bayes variants, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and SVC) using both vectorization methods, focusing on precision as a key metric due to the imbalanced nature of the dataset.
* The Voting Classifier with soft voting, using a combination of Logistic Regression, Multinomial Naive Bayes, Bernoulli Naive Bayes, XGBoost, and Random Forest, achieved the highest precision (1.0) and a high accuracy (0.9739), making it the most effective model for this spam detection task based on THE experiments.
* The Stacking Classifier did not provide a better result in this case.
* In conclusion, the soft voting classifier appears to be the most robust model for identifying spam messages in this dataset, achieving perfect precision on the test set while maintaining high accuracy.
