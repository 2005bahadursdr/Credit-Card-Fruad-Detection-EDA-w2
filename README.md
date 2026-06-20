# Credit Card Fraud Detection EDA

This project explores an anonymized Kaggle dataset of credit card transactions to detect fraudulent activities using various EDA techniques and machine learning models.

## Key EDA Techniques Addressed
- **Time-Series Analysis**: Fraud over time/hours.
- **Correlation & Heatmaps**: Relationships between anonymized features.
- **Geospatial Visualization**: If location mapping is relevant.
- **Outlier & Anomaly Detection**: Statistical bound checks.
- **Histograms, Boxplots, KDE**: Distribution comparisons.
- **PCA & Clustering**: Unsupervised exploratory segmentation.
- **Demographic & Economic Stratification**: Analyzing transaction classes.

## Project Structure

```
credit-card-fraud-detection/
├── data/                  # Contains raw and processed data
├── notebooks/             # Jupyter notebooks for stepwise EDA and modeling
├── src/                   # Source code modules (preprocessing, feature engineering, models)
├── models/                # Saved serialized models (e.g., .pkl)
├── reports/               # HTML/TXT reports and figures
├── main.py                # Streamlit Dashboard application
└── requirements.txt       # Python dependencies
```

## Running the Dashboard

To run the interactive EDA dashboard:

```bash
pip install -r requirements.txt
streamlit run main.py
```
