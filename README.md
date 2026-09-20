# Financial News Sentiment Classification Using Deep Learning

A deep learning-based NLP project that classifies finance-related tweets and financial text into **Bearish, Bullish, and Neutral** sentiment using RNN, LSTM, and GRU models.

The **GRU model** was selected for deployment through a Streamlit application.

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

## 📥 Trained Model Files

The trained GRU model and supporting files are available on Google Drive.

[Download Model Files – Google Drive](https://drive.google.com/drive/folders/1rd7bhQi1sVMX-1bU3qI-jX4InZ3KwbQD?usp=sharing)

Download the following four files:

```text
gru_model.pth
word_to_index.pkl
label_mapping.pkl
gru_config.pkl
```

Place all four files in the **same folder as `app.py`**.

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

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Financial-News-Sentiment-Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Model Files

Download the four trained model files from the Google Drive link above and place them in the same folder as `app.py`.

### 4. Run Streamlit

```bash
streamlit run app.py
```

Open the local URL shown in the term
