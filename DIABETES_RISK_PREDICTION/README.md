## DIABETES_RISK_PREDICTION
Data Loading, Understanding Data, Data Preprocessing, Feature Engineering, Model Building, Model Evaluation

The glucose features show normal distributions with slight skewness. The target variable 'Outcome' is imbalanced with 65% non-diabetic and 35% diabetic cases. Feature engineering created new features like glucose categories and BMI categories. Correlation analysis revealed strong relationships between glucose, BMI, and diabetes risk.

This project focuses on predicting diabetes risk using a dataset containing glucose levels and other health metrics. The goal is to build a machine learning model that can accurately classify individuals as diabetic or non-diabetic based on various features.

LightGBM achieved the highest accuracy of 0.9143 with AUC of 0.967. Other models like Random Forest (accuracy 0.896), XGBoost (accuracy 0.907), and CatBoost (accuracy 0.909) were also evaluated.

!feature_importance_and_confusion_matrix.png !feature_pairplot_risk.png !glucose_correlation_heatmap.png !glucose_features_boxplots.png !glucose_features_distributions.png !model_performance_comparison.png !risk_distribution_analysis.png !roc_curves_comparison.png !shap_summary_plot.png