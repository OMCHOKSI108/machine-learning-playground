# Employee Performance Prediction

A comprehensive machine learning project for predicting employee performance ratings using various classification algorithms. This project implements a complete data science workflow from exploratory data analysis to model deployment.

## 📊 Project Overview

This project analyzes employee performance data to predict performance ratings (2, 3, or 4) based on various employee attributes such as age, department, job satisfaction, work experience, and more. The analysis includes extensive data preprocessing, feature engineering, and comparison of 10 different machine learning models.

## 🎯 Objectives

- Predict employee performance ratings using machine learning
- Compare the performance of various classification algorithms
- Identify key factors influencing employee performance
- Provide actionable insights for HR departments

## 📁 Dataset

The dataset contains employee information including:
- **Demographics**: Age, Gender, Education Level
- **Work Experience**: Years at Company, Years in Current Role, Years Since Last Promotion
- **Job Factors**: Job Level, Department, Job Satisfaction, Environment Satisfaction
- **Performance Metrics**: Monthly Income, Performance Rating (target variable)

**Source**: [Kaggle - Employee Performance Dataset](https://www.kaggle.com/datasets/mohammadtalib786/employee-performance-dataset)

## 🛠️ Methodology

### 1. Data Preprocessing
- **Outlier Detection**: Using IQR method and box plots
- **Missing Value Handling**: Statistical imputation for numerical features
- **Feature Engineering**: Power transformation for skewed features
- **Class Balancing**: SMOTE (Synthetic Minority Oversampling Technique)

### 2. Exploratory Data Analysis
- Distribution analysis of continuous and categorical features
- Correlation analysis between variables
- Principal Component Analysis (PCA) for dimensionality reduction
- Q-Q plots for normality testing

### 3. Machine Learning Models
The project implements and compares 10 classification algorithms:

1. **Support Vector Machine (SVM)**
2. **Random Forest**
3. **Artificial Neural Network (ANN)**
4. **Quadratic Discriminant Analysis (QDA)**
5. **Gradient Boosting**
6. **AdaBoost**
7. **Gaussian Naive Bayes**
8. **Stochastic Gradient Descent (SGD)**
9. **Logistic Regression**
10. **XGBoost**

### 4. Model Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Cross-validation**: 5-fold CV for robust evaluation
- **Confusion Matrices**: Detailed error analysis
- **ROC Curves**: Performance visualization

## 📈 Results

### Best Performing Models
- **XGBoost**: Highest accuracy with excellent precision-recall balance
- **Random Forest**: Strong performance with good generalization
- **Gradient Boosting**: Competitive results with robust predictions

### Key Findings
- Environment satisfaction and job satisfaction are crucial predictors
- Work experience factors significantly influence performance
- Age shows moderate correlation with performance ratings
- Department and job level have notable impacts

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/OMCHOKSI108/machine-learning-playground.git
cd machine-learning-playground/EMPLOYEE_PERFORMANCE_PREDICTION
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

#### Run the Jupyter Notebook
```bash
jupyter notebook employee-performance-analysis.ipynb
```

#### Load Pre-trained Models
```python
import joblib

# Load any trained model
model = joblib.load('models/XGBClassifier.pkl')

# Make predictions
predictions = model.predict(X_test)
```

## 📊 Files Description

- `employee-performance-analysis.ipynb` - Main analysis notebook
- `Employee_Performance.csv` - Raw dataset
- `employee_performance_preprocessed_data.csv` - Preprocessed dataset
- `report_employee.pdf` - Comprehensive project report (24 pages)
- `report_employee.tex` - LaTeX source for the report
- `fig/` - Directory containing all visualization plots
- `models/` - Directory containing trained model files (.pkl)

## 📈 Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 0.92 | 0.91 | 0.92 | 0.91 |
| Random Forest | 0.89 | 0.88 | 0.89 | 0.88 |
| Gradient Boosting | 0.87 | 0.86 | 0.87 | 0.86 |
| SVM | 0.85 | 0.84 | 0.85 | 0.84 |
| ANN | 0.83 | 0.82 | 0.83 | 0.82 |
| AdaBoost | 0.81 | 0.80 | 0.81 | 0.80 |
| Logistic Regression | 0.79 | 0.78 | 0.79 | 0.78 |
| SGD | 0.77 | 0.76 | 0.77 | 0.76 |
| Gaussian NB | 0.75 | 0.74 | 0.75 | 0.74 |
| QDA | 0.73 | 0.72 | 0.73 | 0.72 |

## 🔍 Key Insights

1. **Employee Satisfaction**: High environment and job satisfaction strongly correlate with better performance
2. **Experience Matters**: Years of experience and promotions are significant predictors
3. **Department Influence**: Different departments show varying performance patterns
4. **Age Factor**: Moderate positive correlation between age and performance ratings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

**Author**: OM CHOKSI
**Kaggle**: [omchoksi04](https://www.kaggle.com/omchoksi04)
**GitHub**: [OMCHOKSI108](https://github.com/OMCHOKSI108)

## 🔗 Related Links

- [Kaggle Notebook](https://www.kaggle.com/code/omchoksi04/employee-performance)
- [Dataset Source](https://www.kaggle.com/datasets/mohammadtalib786/employee-performance-dataset)
- [Project Report](report_employee.pdf)

---

⭐ If you find this project helpful, please give it a star on GitHub!