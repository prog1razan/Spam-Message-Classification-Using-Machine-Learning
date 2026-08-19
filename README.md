# Spam Message Classification Using Machine Learning

A **Natural Language Processing (NLP)** project that classifies text messages as **spam** or **ham** using Python and machine learning.

## 📌 Overview

This project explores a text message dataset and builds machine learning models to classify messages as:

* **ham** — a normal message
* **spam** — an unwanted promotional or scam message

The notebook demonstrates the complete workflow of a basic text classification project, including data exploration, text cleaning, TF-IDF feature extraction, model training, model comparison, evaluation, and testing with new messages.

## 📊 Dataset

The dataset contains **193 text messages** with three columns:

* `label` — message category (`ham` or `spam`)
* `label_num` — numerical label (`0` for ham, `1` for spam)
* `message` — the text message

The dataset contains:

* **166 ham messages**
* **27 spam messages**
* **0 missing values**
* **0 duplicated rows**
  A `message_length` column is also created to analyze the number of characters in each message. The average message length is approximately **49 characters for ham** and **85 characters for spam**.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / Google Colab

## 🔍 Topics Covered

The notebook demonstrates:

* Loading a text dataset using Pandas
* Exploring dataset shape, columns, and data types
* Checking class distribution
* Checking missing and duplicated values
* Analyzing message length
* Cleaning text data
* Converting text to lowercase
* Removing punctuation and numbers
* Removing extra spaces
* Exploratory data analysis and visualization
* Finding common words in spam and ham messages
* Splitting data into training and testing sets
* Converting text into numerical features using **TF-IDF**
* Training multiple machine learning classification models
* Comparing model accuracy
* Evaluating the best-performing model
* Using a confusion matrix
* Generating a classification report
* Predicting new custom messages

## 🤖 Machine Learning Models

The project compares five classification algorithms:

1. **Multinomial Naive Bayes**
2. **Logistic Regression**
3. **Linear Support Vector Machine (Linear SVM)**
4. **Decision Tree**
5. **Random Forest**

The dataset is split using an **80/20 train-test split**, with `random_state=42` and stratification. The messages are then transformed into numerical features using `TfidfVectorizer`.

## 📈 Model Evaluation

The models are compared based on their classification accuracy. The best-performing model is then evaluated using:

* Confusion matrix
* Precision
* Recall
* F1-score
* Classification report

In the notebook's evaluation output, the selected **Decision Tree** model achieved **95% accuracy** on the test set. For the spam class, it achieved **1.00 precision**, **0.60 recall**, and **0.75 F1-score**.

## 🧪 Custom Message Testing

The trained model is also tested on new messages to determine whether they are spam or ham.

Examples include:

* Lottery and prize messages
* Gift card offers
* Bank security messages
* Academy acceptance messages
* Casual conversations

The new messages are cleaned and transformed using the same fitted TF-IDF vectorizer before making predictions.

## 📁 Project Structure

```text
Spam-Message-Classification/
│
├── Spam_Message_Classification_Text_ML_Project.ipynb
├── real_sms_spam_dataset_sample.csv
└── README.md
```

## 👩🏻‍💻 Author

**Razan Alothaim**
