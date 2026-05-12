# Credit Card Fraud Detection

A Streamlit web application for detecting fraudulent credit card transactions using machine learning models.

## Features

- Interactive web interface for inputting transaction features
- Real-time fraud prediction using Random Forest classifier
- Displays prediction probability

## Dataset

The model is trained on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle, which contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly imbalanced with 492 frauds out of 284,807 transactions.

## Model

- **Algorithm**: Random Forest Classifier
- **Features**: Time, V1-V28 (PCA transformed features), Amount
- **Training**: Used SMOTE for handling class imbalance

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` and enter the transaction features to get a fraud prediction.

## Deployment

This app can be deployed on Streamlit Cloud:

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy the app

## Project Structure

```
credit-card-fraud-detection/
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
├── model/                # Trained models
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
└── notebook/             # Jupyter notebook with model training
    └── credit_card_fraud_detection.ipynb
```

## License

This project is for educational purposes only.