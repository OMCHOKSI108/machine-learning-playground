# Memory Test on Drugged Islanders — Cognitive Recall Study

## Classification of Anti-Anxiety Medicine Effects on Memory Recall under Happy vs Sad Priming

This project investigates the effects of anti-anxiety medications on short-term memory recall, factoring in emotional priming (happy vs. sad states) and demographic variables. The study involves 198 participants divided into three drug groups: Alprazolam (A), Triazolam (T), and a placebo (S), with varying dosages and emotional priming conditions. Memory scores were assessed before and after a 7-day treatment period to evaluate cognitive impacts.

The analysis reveals that memory decline is a consistent outcome post-treatment, but its severity depends on drug type, emotional priming, and age. Alprazolam shows the least memory impairment, while Triazolam causes the most significant decline. Positive emotional priming enhances memory retention, whereas negative priming exacerbates loss. Age negatively correlates with memory performance, with older adults being more vulnerable.

## Steps

1. **Data Loading and Exploration**: Imported the dataset, performed basic information checks, and descriptive statistics to understand the data structure and distributions.
2. **Exploratory Data Analysis (EDA)**: Created visualizations including bar plots, pie charts, pair plots, scatter matrices, box plots, heatmaps, and correlation analyses to identify patterns in memory scores, drug distributions, age effects, and emotional priming influences.
3. **Data Preprocessing**: Cleaned the data by handling missing values, encoding categorical variables (e.g., drug types, age ranges), and scaling numerical features using MinMaxScaler for model compatibility.
4. **Predictive Analysis**: Trained and evaluated multiple machine learning models (e.g., KNN, Logistic Regression, SVM, Random Forest, Gradient Boosting) to predict memory improvement (Diff > 0). Assessed performance using F1 scores, confusion matrices, ROC-AUC curves, and feature importance.

## Insights

- **Drug Effects**: Alprazolam (A) provides the best memory retention with higher positive Diff values, followed by placebo (S) with moderate decline, and Triazolam (T) with the strongest memory deterioration.
- **Emotional Priming**: Happy priming significantly boosts memory scores across all drugs, while sad priming leads to greater and more variable memory loss.
- **Age Impact**: Memory performance declines with age, with younger participants (18-35) showing stronger resilience and older adults (65+) exhibiting uniform low performance.
- **Dosage Correlation**: Dosage shows weak correlation with memory changes, indicating that drug type and emotional state are more influential.
- **Model Insights**: Memory-related features (e.g., Diff, Mem_Score_Before) are the strongest predictors, outweighing demographic variables like age and dosage.

## Results

The predictive models achieved varying F1 scores, with Logistic Regression performing best at 0.7778, followed by AdaBoost (0.7692) and SVM (0.7083). The study confirms that memory decline is global but modulated by drug type, emotional priming, and age. Positive priming protects memory, while negative priming worsens impairment, especially in older participants. Alprazolam is recommended for better cognitive outcomes in positive emotional contexts.

## Images

<img src="fig/distribution_of_drugs.png" alt="Distribution of Drugs">
<img src="fig/distribution_of_drug_pie.png" alt="Distribution of Drug Pie">
<img src="fig/pairplot.png" alt="Pairplot">
<img src="fig/scatter_matrix.png" alt="Scatter Matrix">
<img src="fig/boxplot_agerange_diff.png" alt="Boxplot AgeRange Diff">
<img src="fig/boxplot_drug_mem_score_after.png" alt="Boxplot Drug Mem Score After">
<img src="fig/heatmap_nulls.png" alt="Heatmap Nulls">
<img src="fig/correlation_heatmap.png" alt="Correlation Heatmap">
<img src="fig/confusion_matrix_knn.png" alt="Confusion Matrix KNN">
<img src="fig/confusion_matrix_logistic.png" alt="Confusion Matrix Logistic">
<img src="fig/confusion_matrix_decision_tree.png" alt="Confusion Matrix Decision Tree">
<img src="fig/confusion_matrix_svm.png" alt="Confusion Matrix SVM">
<img src="fig/confusion_matrix_random_forest.png" alt="Confusion Matrix Random Forest">
<img src="fig/confusion_matrix_extra_trees.png" alt="Confusion Matrix Extra Trees">
<img src="fig/confusion_matrix_adaboost.png" alt="Confusion Matrix AdaBoost">
<img src="fig/confusion_matrix_gradient_boosting.png" alt="Confusion Matrix Gradient Boosting">
<img src="fig/feature_importance_random_forest.png" alt="Feature Importance Random Forest">
<img src="fig/confusion_matrix_extra_trees_full.png" alt="Confusion Matrix Extra Trees Full">
<img src="fig/feature_importance_extra_trees.png" alt="Feature Importance Extra Trees">
<img src="fig/confusion_matrix_gradient_boosting_full.png" alt="Confusion Matrix Gradient Boosting Full">
<img src="fig/feature_importance_gradient_boosting.png" alt="Feature Importance Gradient Boosting">
<img src="fig/confusion_matrix_voting_classifier.png" alt="Confusion Matrix Voting Classifier">
<img src="fig/model_f1_score_comparison.png" alt="Model F1 Score Comparison">
<img src="fig/roc_auc_curves_all_models.png" alt="ROC AUC Curves All Models">
<img src="fig/best_model_roc_curve.png" alt="Best Model ROC Curve">
<img src="fig/permutation_feature_importance.png" alt="Permutation Feature Importance"></content>
<parameter name="filePath">d:\WORKSPACE\MLDLC\machine-learning-playground\ANTI_ANXIETY_MEMORY_STUDY\README.md