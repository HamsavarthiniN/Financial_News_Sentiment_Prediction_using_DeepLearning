# 📈 Financial News Sentiment Classification Using Deep Learning

## 📌 Project Overview

This project builds a **sentiment classification system for finance-related tweets** using deep learning.

The system classifies financial text into three sentiment categories:

* 🔴 **Bearish**
* 🟢 **Bullish**
* ⚪ **Neutral**

The project implements and compares three recurrent neural network architectures:

* Simple RNN
* LSTM
* GRU

A traditional **TF-IDF + Logistic Regression** model was also developed as a baseline for comparison.

The **GRU model achieved the best performance among the deep learning models** and was selected for deployment using Streamlit.

> **Note:** BERT fine-tuning was an optional extension in the project requirements and was not implemented in this project.

---

## 🎯 Objectives

* Perform sentiment classification on financial tweets.
* Preprocess and clean financial text data.
* Convert text into numerical sequences suitable for deep learning.
* Handle class imbalance during model training.
* Implement RNN, LSTM, and GRU models.
* Compare model performance using multiple evaluation metrics.
* Select the best-performing deep learning model.
* Deploy the selected GRU model using Streamlit.

---

## 📊 Dataset

The project uses the **Financial News Sentiment** dataset containing finance-related tweets.

The dataset contains three sentiment classes:

| Label | Sentiment |
| ----: | --------- |
|     0 | Bearish   |
|     1 | Bullish   |
|     2 | Neutral   |

### Class Distribution

The training dataset contains:

| Sentiment | Samples |
| --------- | ------: |
| Bearish   |   1,442 |
| Bullish   |   1,923 |
| Neutral   |   6,178 |

The dataset is imbalanced, with Neutral being the majority class. Therefore, class-weighted loss was used during deep learning model training.

---

## 🔄 Project Workflow

```text
Financial Tweets
       ↓
Text Preprocessing
       ↓
Clean Text
       ↓
Tokenization
       ↓
Vocabulary Creation
       ↓
Integer Sequences
       ↓
Padding
       ↓
Class-Weighted Training
       ↓
┌───────────────┬───────────────┬───────────────┐
│     RNN       │     LSTM      │      GRU       │
└───────────────┴───────────────┴───────────────┘
       ↓
Model Evaluation
       ↓
GRU Selected
       ↓
Streamlit Deployment
       ↓
Bearish / Bullish / Neutral
```

---

## 🧹 Text Preprocessing

The dataset was prepared using a text-cleaning pipeline before sequence generation.

The preprocessing process included operations such as:

* Lowercasing
* Removing unnecessary spaces
* Unicode normalization
* Removing URLs
* Removing email addresses
* Removing mentions
* Removing unnecessary special characters
* Tokenization
* Stopword handling
* Removing very short words
* Lemmatization

The resulting `clean_text` column was used as the input for the deep learning models.

---

## 🔢 Sequence Preparation

The cleaned text was converted into integer sequences using a vocabulary.

Special tokens were used:

```text
<PAD> → Padding token
<UNK> → Unknown token
```

### Sequence Information

```text
Vocabulary Size : 20,933
Maximum Length  : 27
Training Samples: 9,543
Testing Samples : 2,388
```

Each text sequence was padded or truncated to a maximum length of **27 tokens**.

---

## 🧠 Deep Learning Models

Three recurrent neural network architectures were implemented.

### 1. Simple RNN

A basic recurrent neural network was implemented as the initial deep learning model.

### 2. LSTM

Long Short-Term Memory was implemented to better handle dependencies in sequential text.

### 3. GRU

Gated Recurrent Unit was implemented as another recurrent architecture.

The GRU achieved the strongest performance among the three deep learning models and was therefore selected for deployment.

---

## ⚖️ Class Imbalance

The dataset contains significantly more Neutral samples than Bearish and Bullish samples.

To reduce the effect of this imbalance, **class-weighted Cross Entropy Loss** was used during model training.

This encourages the model to pay more attention to the minority classes.

---

## 📈 Model Performance

The models were evaluated using:

* Accuracy
* Precision
* Recall
* Macro F1-score

### Deep Learning Model Comparison

| Model   |   Accuracy |  Precision |     Recall |   Macro F1 |
| ------- | ---------: | ---------: | ---------: | ---------: |
| RNN     |     35.43% |     36.57% |     37.80% |     32.62% |
| LSTM    |     59.55% |     60.64% |     57.26% |     53.29% |
| **GRU** | **75.50%** | **71.64%** | **61.61%** | **63.90%** |

### Traditional Baseline

A TF-IDF + Logistic Regression model was also implemented as a traditional machine-learning baseline.

| Model                        | Accuracy | Macro F1 |
| ---------------------------- | -------: | -------: |
| TF-IDF + Logistic Regression |   80.61% |   74.50% |
| GRU                          |   75.50% |   63.90% |

The Logistic Regression model performed better overall as a traditional baseline, while **GRU was the strongest deep learning model** and was selected for the final application.

---

## 🏆 Selected Model

### GRU — Gated Recurrent Unit

The GRU model was selected for deployment because it achieved the best results among the implemented deep learning architectures.

Final GRU performance:

```text
Accuracy  : 75.50%
Precision : 71.64%
Recall    : 61.61%
Macro F1  : 63.90%
```

---

## 🌐 Streamlit Application

The trained GRU model was integrated into a Streamlit application.

The application allows the user to:

1. Enter financial news or a financial tweet.
2. Convert the input into a sequence using the saved vocabulary.
3. Pass the sequence through the trained GRU model.
4. Predict the sentiment.
5. Display the prediction confidence.
6. Display probabilities for all three sentiment classes.

### Example

```text
Input:
The company reported record profits and strong revenue growth.

Prediction:
Bullish 📈

Confidence:
XX.XX%
```

The actual confidence is generated by the trained model.

---

## 📁 Project Structure

```text
Financial-News-Sentiment-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── Financial_News_Sentiment_Classification.ipynb
│
└── models/
    ├── gru_model.pth
    ├── word_to_index.pkl
    ├── label_mapping.pkl
    └── gru_config.pkl
```

---

## 💾 Model Files

The trained model and supporting files are:

```text
gru_model.pth
word_to_index.pkl
label_mapping.pkl
gru_config.pkl
```

### File Description

| File                | Purpose                                   |
| ------------------- | ----------------------------------------- |
| `gru_model.pth`     | Trained GRU model weights                 |
| `word_to_index.pkl` | Vocabulary-to-index mapping               |
| `label_mapping.pkl` | Numeric label-to-sentiment mapping        |
| `gru_config.pkl`    | Model configuration and sequence settings |

If the model files are hosted separately on Google Drive, add the sharing link here:

**[Download Trained GRU Model Files](PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE)**

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* PyTorch
* Scikit-learn
* Streamlit
* Google Colab
* Google Drive
* GitHub

---

## 🚀 Running the Streamlit Application

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Financial-News-Sentiment-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the model files

Download the trained model files and place them inside:

```text
models/
```

### 4. Run Streamlit

```bash
streamlit run app.py
```

The application will open in the browser.

---

## 🔮 Future Enhancement

The project requirement also provides an optional **BERT fine-tuning** extension.

BERT was **not implemented in this project**.

Possible future improvements include:

* Fine-tuning a BERT-based model.
* Using pretrained financial language models.
* Improving text preprocessing for financial terminology.
* Hyperparameter tuning.
* Increasing training data.
* Deploying the application to a cloud platform.

---

## 📌 Conclusion

This project demonstrates a complete deep learning pipeline for financial sentiment classification.

RNN, LSTM, and GRU architectures were implemented and compared. Among the deep learning approaches, **GRU achieved the best performance** and was selected for deployment.

The final GRU model was integrated into a Streamlit application that provides real-time classification of financial text into:

**Bearish 🔴 | Bullish 🟢 | Neutral ⚪**

The project fulfills the mandatory deep learning component of the assignment, while BERT fine-tuning remains an optional future enhancement.

## 📌 Drive link

https://drive.google.com/drive/folders/1rd7bhQi1sVMX-1bU3qI-jX4InZ3KwbQD?usp=sharing
