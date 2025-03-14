# 📧 Spam Email Classifier 🔍  
![Spam Detector](https://img.shields.io/badge/Spam-Detector-green?style=for-the-badge)  
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge)  
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Naive%20Bayes-orange?style=for-the-badge)  
![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-red?style=for-the-badge)  

---

## 🚀 **Project Overview**  

This project is a **Spam Email Classifier** using **Naïve Bayes** and **TF-IDF vectorization** to detect spam emails with high accuracy. The model is trained on a dataset of spam and ham (non-spam) emails and can classify new emails as **Spam** or **Not Spam**.  

### 🔹 **Tech Stack Used**
- 🐍 Python (pandas, NumPy, scikit-learn)
- 🤖 Machine Learning (Naïve Bayes, TF-IDF Vectorization)
- 📊 Data Visualization (matplotlib, seaborn)
- 🔬 Model Evaluation (Classification Report, Accuracy Score)
- 🏢 Deployment Ready (Pickle for Model Storage)

---

## 📜 **Table of Contents**
- [📂 Dataset](#dataset)
- [⚙️ Installation](#installation)
- [📊 Exploratory Data Analysis](#exploratory-data-analysis)
- [🛠️ Model Training & Evaluation](#model-training--evaluation)
- [💾 Save & Load Model](#save--load-model)
- [📩 Test with New Emails](#test-with-new-emails)
- [📸 Screenshots & Graphs](#screenshots--graphs)
- [🛠️ Contributing](#contributing)
- [📜 License](#license)

---

## 📂 **Dataset**  
The dataset used consists of labeled email messages categorized as `spam` or `ham`.  

| Column Name  | Description  |
|-------------|-------------|
| `Unnamed: 0` | Index |
| `label` | Spam or Ham (Non-Spam) |
| `text` | Email content |
| `label_num` | 1 for Spam, 0 for Ham |

**📌 Sample Data:**  
```
   Unnamed: 0    label   text                                             label_num  
0         605    ham    Subject: Enron methanol ...                          0  
1        2349    ham    Subject: HPL Nom for January 9 ...                  0  
2        3624    ham    Subject: Neon retreat ...                           0  
3        4685   spam    Subject: Photoshop, Windows, Office ...            1  
4        2030    ham    Subject: Re: Indian Springs ...                     0  
```

---

## ⚙️ **Installation**  

Clone the repository and install dependencies:  
```bash
git clone https://github.com/yourusername/spam-email-classifier.git
cd spam-email-classifier
pip install -r requirements.txt
```

### ✅ **Run on Google Colab**
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 📊 **Exploratory Data Analysis**  

### **Spam vs. Ham Distribution**
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='label', palette="coolwarm")
plt.title("Spam vs. Ham Distribution")
plt.show()
```
📌 **Output:**  
![Spam Ham Distribution](https://your-image-link.com/spam-ham-chart.png)

---

## 🛠️ **Model Training & Evaluation**  

1️⃣ **Preprocessing:**  
- **Text Cleaning** (Removing stopwords, special characters)
- **Tokenization & Vectorization** (TF-IDF)  
- **Splitting Data** (80% Train, 20% Test)

2️⃣ **Training Model using Naïve Bayes:**  
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate Model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))
```
📌 **Expected Accuracy:**  
🎯 **92.46%**

---

## 💾 **Save & Load Model**
Save trained model and vectorizer:
```python
import pickle

pickle.dump(model, open("spam_classifier.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
```
Load trained model and vectorizer:
```python
loaded_model = pickle.load(open("spam_classifier.pkl", "rb"))
loaded_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
```

---

## 📩 **Test with New Emails**
```python
new_message = ["Congratulations! You've won a free trip to Hawaii. Click here to claim."]
new_message_tfidf = loaded_vectorizer.transform(new_message)
prediction = loaded_model.predict(new_message_tfidf)

print("Spam" if prediction[0] == 1 else "Not Spam")
```
📌 **Output:**  
🛑 **Spam!** 🚨  

---

## 📸 **Screenshots & Graphs**  

| **Spam vs Ham Distribution** | **TF-IDF Feature Importance** |
|-----------------------------|------------------------------|
| ![Spam Ham Graph](https://your-image-link.com/spam-ham.png) | ![TF-IDF](https://your-image-link.com/tfidf.png) |

---

## 🛠️ **Contributing**
Contributions are welcome! To contribute:  
1. Fork the repository  
2. Create a new branch (`feature-branch`)  
3. Commit your changes  
4. Push and submit a PR 🎉  

---

## 📜 **License**  
This project is licensed under the **MIT License**.  

📧 **Developed by [Your Name](https://github.com/yourusername)**  
⭐ Star this repository if you found it useful! 🚀
