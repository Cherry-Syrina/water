# 💧 AI-Based Water Quality Prediction System

A Machine Learning-powered web application that predicts water potability using physicochemical parameters — supporting real-time aquatic ecosystem monitoring and environmental health assessment.

## 🌊 Problem Statement

Access to safe drinking water is a critical global challenge, especially in wetland and riverine ecosystems impacted by climate change and agricultural runoff. Traditional water quality testing is expensive and slow. This system provides instant AI-driven predictions aligned with WHO standards.

## 🚀 Live Demo

[➡ Open App](https://your-streamlit-url.streamlit.app)

## 🗂️ Project Structure

```
water-quality-prediction/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── data/
│   ├── __init__.py
│   └── generate_data.py    # Synthetic dataset generation (WHO-based)
├── model/
│   ├── __init__.py
│   └── train_model.py      # Model training, evaluation pipeline
└── utils/
    ├── __init__.py
    └── predictor.py        # Inference, WHO validation, recommendations
```

## 🤖 Technical Approach

| Component | Detail |
|---|---|
| Algorithm | Gradient Boosting Classifier |
| Features | 8 physicochemical parameters |
| Preprocessing | StandardScaler normalization |
| Validation | 80/20 stratified train-test split |
| Metrics | Accuracy, ROC-AUC, F1-score |

## 📊 Parameters Analyzed

| Parameter | WHO Safe Range |
|---|---|
| pH | 6.5 – 8.5 |
| Dissolved Oxygen | > 6.5 mg/L |
| Turbidity | < 4 NTU |
| Temperature | 15 – 27°C |
| Nitrate | < 10 mg/L |
| BOD | < 3 mg/L |
| Conductivity | 200 – 600 µS/cm |
| Coliform | < 50 CFU/100mL |

## ⚙️ Run Locally

```bash
git clone https://github.com/your-username/water-quality-prediction.git
cd water-quality-prediction
pip install -r requirements.txt
streamlit run app.py
```

## 🌍 Applications

- Real-time monitoring of rivers, wetlands, and lakes
- Rural drinking water safety assessment
- Climate change impact on aquatic ecosystem prediction
- Early warning systems for ecosystem degradation
- Fisheries and cold-chain water quality monitoring

## 👩‍💻 Author

**Sushma Shukla** | Integrated M.Tech Software Engineering | VIT Vellore  
[GitHub](https://github.com/Cherry-Syrina) | [Email](mailto:sushma.shukla3011@gmail.com)
