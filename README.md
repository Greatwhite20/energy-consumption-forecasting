# ⚡ Household Energy Consumption Forecasting

A machine learning project to predict household electricity consumption using historical power usage data. Built as part of my Data Science learning journey at **MNNIT Allahabad**.

---

## 📌 Problem Statement

Energy forecasting is critical for power grid management, cost reduction, and sustainability planning. This project builds ML models to predict **Global Active Power (kW)** consumption based on time-based and electrical features.

---

## 📊 Dataset

**UCI Machine Learning Repository — Individual Household Electric Power Consumption**

- ~2 million minute-level readings over 4 years (2006–2010)
- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
- Download the file and place it at: `data/household_power_consumption.txt`

**Key Features:**
| Feature | Description |
|---|---|
| Global_active_power | Total active power consumed (kW) — **target** |
| Voltage | Minute-averaged voltage (V) |
| Global_intensity | Household current intensity (A) |
| Sub_metering_1 | Kitchen appliances energy |
| Sub_metering_2 | Laundry room energy |
| Sub_metering_3 | Electric water heater & AC energy |

---

## 🧠 Models Used

| Model | Description |
|---|---|
| Linear Regression | Baseline model |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosting — typically best performer |

---

## 📁 Project Structure

```
energy-forecasting/
│
├── data/
│   └── household_power_consumption.txt   ← Download from UCI
│
├── outputs/
│   ├── eda_plots.png
│   ├── model_results.png
│   └── feature_importance.png
│
├── energy_forecasting.py    ← Main script
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/energy-forecasting.git
cd energy-forecasting
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
Go to [this link](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption), download the zip, extract and place the `.txt` file in the `data/` folder.

### 4. Run the project
```bash
python energy_forecasting.py
```

---

## 📈 Results

After training, all plots are saved to the `outputs/` folder:

- **EDA Plots** — hourly trends, monthly patterns, weekday vs weekend, correlation heatmap
- **Actual vs Predicted** — visual comparison of best model
- **Feature Importance** — which features matter most

Typical R² scores:
- Linear Regression: ~0.85
- Random Forest: ~0.97
- Gradient Boosting: ~0.96

---

## 🔧 Tech Stack

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 💡 Key Learnings

- Time-series feature engineering (hour, month, weekday)
- Handling large datasets with resampling
- Comparing regression models using MAE, RMSE, R²
- Feature importance analysis with Random Forest

---

## 👤 Author

**Your Name**
B.Tech Electrical Engineering | MNNIT Allahabad
[LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)
