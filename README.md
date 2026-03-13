# FraudShield — Cloud-Based Fraud Detection System

A machine-learning-powered web application that analyses financial transactions
in real time and predicts whether they are **Fraudulent** or **Legitimate**.

---

## Folder Structure

```
fraud_detection/
│
├── train_model.py          ← Generate dataset, train & save ML model
├── app.py                  ← Flask backend + /predict API endpoint
├── requirements.txt        ← Python dependencies
├── Procfile                ← For Render / Heroku deployment
│
├── model_data/             ← Created automatically by train_model.py
│   ├── fraud_model.pkl     ← Trained Random Forest classifier
│   ├── scaler.pkl          ← Fitted StandardScaler
│   └── features.pkl        ← Feature name order (ensures consistency)
│
├── templates/
│   └── index.html          ← Single-page frontend (Jinja2)
│
└── static/
    ├── css/
    │   └── style.css       ← Dark industrial theme, fully responsive
    └── js/
        └── main.js         ← Form handling, validation, result rendering
```

---

## Quick Start (Local)

```bash
# 1. Clone / create the project folder
cd fraud_detection

# 2. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (creates model_data/*.pkl)
python train_model.py

# 5. Start the Flask dev server
python app.py
# → Open http://localhost:5000
```

---

## Machine Learning Details

| Item | Detail |
|------|--------|
| Algorithm | Random Forest Classifier |
| Dataset | Synthetic (10 000 transactions, 5% fraud rate) |
| Features | 10 risk signals (see below) |
| Test Accuracy | ~99.9% |
| Test F1 (fraud) | ~0.995 |

### Features Used

| Feature | Description |
|---------|-------------|
| `amount` | Transaction value in USD |
| `hour_of_day` | Hour the transaction occurred (0–23) |
| `day_of_week` | Day of the week (0 = Mon, 6 = Sun) |
| `transaction_type` | 0 Online / 1 ATM / 2 POS / 3 Wire |
| `merchant_category` | Category code 0–9 |
| `distance_from_home` | Estimated km from account holder's home |
| `num_prev_transactions` | Number of transactions in past 30 days |
| `account_age_days` | Age of the account in days |
| `failed_attempts` | Number of failed authentication attempts |
| `is_international` | 1 if cross-border, 0 if domestic |

---

## API

### `POST /predict`

**Request** (JSON body):
```json
{
  "amount": 4500,
  "hour_of_day": 2,
  "day_of_week": 5,
  "transaction_type": 3,
  "merchant_category": 2,
  "distance_from_home": 850,
  "num_prev_transactions": 1,
  "account_age_days": 30,
  "failed_attempts": 2,
  "is_international": 1
}
```

**Response**:
```json
{
  "prediction": 1,
  "label": "Fraud",
  "probability": 94.3,
  "risk_level": "High",
  "features_used": ["amount", "hour_of_day", ...]
}
```

### `GET /health`
Returns `{"status": "ok", "model_loaded": true}`.

---

## Cloud Deployment

### Render (recommended — free tier)

1. Push project to GitHub.
2. Create a new **Web Service** on [render.com](https://render.com).
3. Build command: `pip install -r requirements.txt && python train_model.py`
4. Start command: `gunicorn app:app`
5. The `Procfile` is already configured.

### Heroku

```bash
heroku create fraudshield-app
git push heroku main
heroku run python train_model.py
heroku open
```

---

## Disclaimer
This project is for educational / demonstration purposes only.
The model is trained on **synthetic data** and should not be used for real
financial decisions without retraining on verified, real-world datasets.
