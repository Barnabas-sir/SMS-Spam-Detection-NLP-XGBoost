# SMS Spam Detection using NLP, XGBoost and Streamlit

## Project Overview

This project develops an end-to-end SMS Spam Detection System using Natural Language Processing (NLP), Machine Learning, Explainable AI (SHAP), and Streamlit deployment. The objective is to automatically classify SMS messages as either **Spam** or **Ham (Legitimate)**.

The project demonstrates the complete machine learning lifecycle, including data preprocessing, exploratory data analysis, feature engineering, model development, hyperparameter tuning, model explainability, and deployment.

---

## Business Problem

Spam messages remain a major communication challenge, often containing fraudulent promotions, phishing attempts, and unwanted advertisements. Manual filtering is inefficient and unreliable, making automated spam detection systems essential for improving user security and communication quality.

This project aims to build a robust classification system capable of accurately identifying spam messages while minimizing false positives.

---

## Dataset

**Dataset:** SMS Spam Collection Dataset

The dataset contains labelled SMS messages categorized as either:

* Ham (Legitimate Messages)
* Spam Messages

### Dataset Summary

* Total Messages: 5,169
* Binary Classification Problem
* Real-world SMS data collected for spam filtering research

---

## Exploratory Data Analysis (EDA)

Several exploratory analyses were conducted to understand message behaviour and spam characteristics.

### Key Findings

* Ham messages significantly outnumber spam messages, indicating class imbalance.
* Spam messages tend to be longer than legitimate messages.
* Spam messages contain more digits, promotional language, and uppercase characters.
* Certain keywords such as **free**, **claim**, **cash**, **prize**, and **reply** frequently appear in spam messages.

### Visualisations

* Class Distribution
* Feature Comparison by Message Type
* Correlation Heatmap
* Word Clouds
* Top 20 Ham Words
* Top 20 Spam Words
* Top 20 Spam Bigrams

---

## Feature Engineering

To enhance predictive performance, several behavioural features were engineered from the message text:

| Feature           | Description                  |
| ----------------- | ---------------------------- |
| Character Count   | Total number of characters   |
| Word Count        | Total number of words        |
| Digit Count       | Number of numeric characters |
| Uppercase Count   | Number of uppercase letters  |
| Exclamation Count | Number of exclamation marks  |

These features were combined with TF-IDF text representations to provide additional information beyond textual content alone.

---

## Text Preprocessing

The following NLP preprocessing steps were applied:

* Lowercasing
* Removal of punctuation
* Removal of special characters
* Stopword removal
* Tokenization
* Lemmatization

The processed text was transformed using **TF-IDF Vectorization**.

---

## Machine Learning Models

Several classification models were trained and evaluated:

* Naive Bayes
* Logistic Regression
* Random Forest
* XGBoost

To address class imbalance, **Stratified Cross Validation** was used throughout model training and evaluation.

---

## Hyperparameter Tuning

The best-performing model, XGBoost, was further optimized using **RandomizedSearchCV** with 5-fold cross-validation.

The tuned model achieved the highest overall performance.

---

## Model Performance

### Tuned XGBoost Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 98.94% |
| Precision | 94.12% |
| Recall    | 97.71% |
| F1 Score  | 95.88% |
| ROC-AUC   | 99.65% |

### Confusion Matrix

* True Negatives: 895
* False Positives: 8
* False Negatives: 3
* True Positives: 128

The model demonstrates excellent capability in identifying spam messages while maintaining a very low false positive rate.

---

## Explainable AI (SHAP)

SHAP (SHapley Additive exPlanations) was used to interpret model predictions and identify the most influential features.

### Key SHAP Insights

The most important features influencing spam classification were:

1. Digit Count
2. Character Count
3. Uppercase Count
4. Word Count
5. Exclamation Count

Among textual features, words such as:

* free
* mobile
* service
* reply
* claim

were strong indicators of spam messages.

This analysis confirmed that engineered behavioural features significantly improved model performance.

---

## Streamlit Deployment

The final model was deployed using Streamlit to provide a simple interactive user interface.

### Application Features

* Real-time SMS classification
* Spam probability score
* User-friendly interface
* Instant predictions using the trained XGBoost model

Users can enter any SMS message and immediately receive a prediction indicating whether the message is Spam or Ham.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* NLTK
* Scikit-Learn
* XGBoost
* SHAP
* Streamlit
* Joblib

---

## Project Structure

```text
SMS-Spam-Detection-NLP-XGBoost/

├── app.py
├── README.md
├── requirements.txt
├── SMS_Spam_Detection.ipynb
├── spam.csv
├── spam_model.pkl
├── tfidf_vectorizer.pkl
├── confusion_matrix.png
├── model_comparison.png
├── shap_bar.png
├── shap_summary.png
└── streamlit_app.png
```

## Future Improvements

* Deep Learning models (LSTM, GRU, Transformers)
* Real-time email spam detection
* Deployment to Streamlit Cloud
* Explainable prediction dashboard
* Multi-language spam detection

---

## Conclusion

This project successfully developed a highly accurate SMS Spam Detection System by combining NLP techniques, feature engineering, XGBoost classification, SHAP explainability, and Streamlit deployment. The tuned XGBoost model achieved an Accuracy of 98.94% and ROC-AUC of 99.65%, demonstrating strong performance in identifying spam messages while maintaining a low false positive rate.

The project showcases practical skills in data preprocessing, machine learning, model optimization, explainable AI, and application deployment, making it a complete end-to-end NLP portfolio project.
<img width="479" height="316" alt="Screenshot 2026-06-12 211853" src="https://github.com/user-attachments/assets/72f53968-be9a-404a-bc39-016d5cb1e0b3" />
<img width="326" height="340" alt="Screenshot 2026-06-12 211908" src="https://github.com/user-attachments/assets/b0d90e9c-a6c8-4a6c-9a08-d12cd451f3c6" />
<img width="377" height="265" alt="Screenshot 2026-06-11 211740" src="https://github.com/user-attachments/assets/71913e17-c6b6-4af7-91e2-fee559e86150" />
<img width="441" height="334" alt="Screenshot 2026-06-11 223924" src="https://github.com/user-attachments/assets/3f1aee1e-58b2-4f11-b563-6584324aeb90" />
<img width="942" height="436" alt="Screenshot 2026-06-12 170914" src="https://github.com/user-attachments/assets/e7b8ec71-f134-445c-8be6-e75a63fce47c" />
<img width="913" height="430" alt="Screenshot 2026-06-12 171110" src="https://github.com/user-attachments/assets/9dc024f3-47a5-41de-a5e0-fe8c1ef9afc5" />
<img width="3002" height="1736" alt="Model Performance Comparison" src="https://github.com/user-attachments/assets/c8fc2a1b-c29d-4d52-98ff-67ae36bd7197" />
