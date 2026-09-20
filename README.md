# Financial News Sentiment Classification Using Deep Learning

A deep learning-based NLP project that classifies finance-related tweets and news text into **Bearish, Bullish, or Neutral** sentiment using RNN, LSTM, and GRU models. The **GRU model** is used for the Streamlit application.

## 🔄 Project Pipeline

```text
Financial Text
      ↓
Text Preprocessing
      ↓
Tokenization
      ↓
Vocabulary Creation
      ↓
Sequence Conversion
      ↓
Padding
      ↓
RNN / LSTM / GRU
      ↓
Model Evaluation
      ↓
GRU Selected
      ↓
Streamlit Deployment
```

📥 Download Trained Model Files

Download the trained GRU model files from Google Drive:

Download Model Files – Google Drive

Download these four files:

gru_model.pth
word_to_index.pkl
label_mapping.pkl
gru_config.pkl

Place all four files in the same folder as app.py.

### Project Structure

```text
Financial-News-Sentiment-Prediction/
│
├── app.py
├── gru_model.pth
├── word_to_index.pkl
├── label_mapping.pkl
├── gru_config.pkl
└── requirements.txt
```

## 🚀 Run the Streamlit Application

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Financial-News-Sentiment-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit

```bash
streamlit run app.py
```

Open the URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## 📊 Output

Enter financial text and the application predicts:

* 📈 Bullish
* 📉 Bearish
* ➖ Neutral

The application also displays the prediction confidence and class probabilities.

## 🛠️ Technologies

**Python • PyTorch • NLTK • Scikit-learn • Streamlit • Google Colab**
