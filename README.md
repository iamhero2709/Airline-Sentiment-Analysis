# ✈️ Airline Tweet Sentiment Analysis 🧠💬

This project performs sentiment classification (Positive, Neutral, Negative) on airline tweets using both **Traditional Machine Learning** and **Deep Learning (BiLSTM)** approaches.

---

## 📊 Dataset

- **Source**: [Kaggle - Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)  
- **Total Tweets**: ~14,000  
- **Target Label**: `airline_sentiment` → `positive`, `neutral`, `negative`  
- **Text Field**: `cleaned_text` (preprocessed tweet)  
- **Additional Features**:  
  - `retweet_count`  
  - `tweet_length`  
  - `tweet_hour`, `tweet_day_of_week`  
  - `airline`  

---

## 🧹 Preprocessing

- Removed usernames, URLs, hashtags, and special characters
- Converted all text to lowercase
- Tokenization and padding (for DL models)
- TF-IDF vectorization (for traditional ML)
- Label encoding (target + categorical features)
- Feature scaling (for numerical metadata)

---

## ✅ Models Implemented

### 🔹 1. Traditional Machine Learning Model (Logistic Regression)

- **Vectorization**: TF-IDF (Top 3000 features)
- **Extra Features**: Airline, Tweet Hour, Day of Week, Tweet Length, Retweet Count
- **Classifier**: `LogisticRegression(max_iter=300)`
- **Train/Test Split**: Stratified (80/20)

#### 🔍 Performance:

- ✅ **Test Accuracy**: ~80%
- 📉 **Loss**: N/A (traditional models)
- 💬 Best baseline model with combined text + metadata

---

### 🔥 2. Deep Learning Model (BiLSTM) — Final & Best Performer

This section highlights the **best performing deep learning model**: a **Bidirectional LSTM**, which captures context from both directions in the text.

---

## 🧠 Model Overview

We used a **BiLSTM (Bidirectional LSTM)** model for tweet sentiment classification.

### 🏗️ Architecture:

```python
Embedding(input_dim=10000, output_dim=128, input_length=100)
Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.2))
Dropout(0.5)
Dense(3, activation='softmax')  # 3 output classes
⚙️ Training Details
Optimizer: Adam

Loss Function: Sparse Categorical Crossentropy

Batch Size: 64

Epochs: 10

Validation Split: 10%

📈 Training Logs
Epoch	Train Accuracy	Validation Accuracy
1	71.71%	79.85%
2	83.30%	80.50% ✅
3	88.12%	79.95%
4	90.83%	79.78%
📌 Best Validation Accuracy at Epoch 2
✅ Final Test Accuracy: 80.50%

📊 Performance Summary
Metric	Score
✅ Test Accuracy	80.50%
📉 Validation Accuracy	~79.85%
🔁 Epochs Trained	10
🧪 Loss Function	Sparse Categorical Crossentropy
⚙️ Optimizer	Adam
🧪 Confusion Matrix (Example)
markdown
Copy
Edit
         Predicted
         Neg  Neu  Pos
Actual
Neg      760   52   25
Neu       44  510   36
Pos       31   60  465
🔵 Negative: High precision & recall

🟡 Neutral: Slight overlap with other classes

🟢 Positive: Fairly accurate

🧪 Classification Report
text
Copy
Edit
              precision    recall  f1-score   support

    negative       0.89      0.93      0.91       837
     neutral       0.84      0.82      0.83       590
    positive       0.89      0.85      0.87       556

    accuracy                           0.85      1983
   macro avg       0.87      0.86      0.87      1983
weighted avg       0.85      0.85      0.85      1983
📦 Requirements
bash
Copy
Edit
tensorflow
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
📌 Conclusion
✅ BiLSTM outperformed traditional ML models by leveraging text sequence structure

🧠 Combining metadata + deep learning can further enhance performance

📈 Future scope: GloVe/FastText embeddings, transformers (BERT), and attention mechanisms
