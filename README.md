# Amazon-sentiment-analysis-capstone
# 📘 Sentiment Analysis for Amazon Reviews (NLP Capstone)
**Author:** Michael Koffie  

This repository contains my capstone project for performing sentiment analysis on Amazon product reviews using Natural Language Processing (NLP) and machine learning. The goal is to automatically classify reviews as positive or negative and provide actionable insights for product and support teams.

---

## 🔍 Project Overview

Companies like Amazon receive millions of product reviews. Manually reading them to find issues is not scalable. This project builds a classification model that:

- Predicts if a review is **positive** or **negative**
- Handles a **heavily imbalanced** dataset (~90% positive, ~10% negative)
- Identifies key text patterns that drive negative sentiment

The final model selected is **Logistic Regression with SMOTE**, which provides a good balance of performance and interpretability.

---

## 📦 Dataset

- **Source:** Kaggle – Amazon Reviews Dataset  
- **Type:** Text reviews + ratings  
- **Target:** Binary sentiment (positive / negative)  
- **Challenge:** Strong class imbalance → required special handling

---

## 🧠 Methods & Tools

**Languages & Libraries:**
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Imbalanced-learn (SMOTE)
- Matplotlib / Seaborn

**Key Steps:**
1. Text preprocessing (cleaning, tokenization, lemmatization, stopword removal)  
2. Feature extraction using **TF-IDF** (unigrams + bigrams)  
3. Additional features: `word_count`, `char_count`  
4. Train/test split and model evaluation  
5. Handling class imbalance with **SMOTE**  
6. Comparing multiple models and selecting the best

---

## 🏆 Models Evaluated

- Logistic Regression (with class weights)
- **Logistic Regression + SMOTE** ✅ *(final model)*
- Linear SVM (with class weights)
- Random Forest (with hyperparameter tuning)
- Multinomial Naive Bayes (text-only)

**Evaluation metrics:**
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Confusion matrix

---

## 📊 Final Model Performance — Logistic Regression + SMOTE

| Metric           | Score   |
|------------------|---------|
| Accuracy         | 0.9257  |
| Macro Precision  | 0.5815  |
| Macro Recall     | 0.5873  |
| Macro F1-score   | 0.5827  |

**Why this model?**

- Best balanced performance across classes  
- Much better at detecting **negative reviews** (minority class)  
- Interpretable coefficients → easy to explain to stakeholders  
- Efficient and scalable for deployment

---

## 📈 Key Insights

From the model and EDA:

- Negative reviews often contain words like **"broken"**, **"defective"**, **"cheap"**, **"doesn't work"**
- Positive reviews often contain **"excellent"**, **"perfect"**, **"great"**, **"durable"**
- The dataset is heavily skewed toward positive reviews
- Text length and structure differ across sentiment classes

These insights can guide product quality improvements and customer support.

---

## 🏢 Business Impact & Recommendations

1. **Automated Monitoring**  
   Use the model to continuously monitor new product reviews and flag items with rising negative sentiment.

2. **Product Quality Improvement**  
   Use the top negative keywords and feature analysis to drive design and quality fixes.

3. **Customer Support Prioritization**  
   Route highly negative reviews to customer service for faster intervention and improved customer satisfaction.

---

## 🚀 Future Work

- Extend to **three classes** (negative / neutral / positive)  
- Try transformer-based models (BERT, RoBERTa)  
- Add topic modeling to find root causes of complaints  
- Deploy as a real-time sentiment API for live review streams

---

## 📁 Repository Structure

```text
/notebooks
    01_EDA.ipynb
    02_Preprocessing_and_Modeling.ipynb
    03_Final_Model_and_Evaluation.ipynb

/report
    Capstone_Final_Report_Michael_Koffie.pdf

/presentation
    Capstone_Submission_Presentation_Michael_Koffie.pptx

/metrics
    model_metrics.csv
    model_metrics.txt
