Classification of Smartphone Addiction Using Supervised Learning
An automated Machine Learning-based classification system designed to identify Problematic Smartphone Use (PSU) in adolescents. This project replaces traditional, subjective paper-based clinical surveys with an objective, data-driven diagnostic pipeline.

🚀 Project Overview
Traditional clinical screening for smartphone addiction relies heavily on self-reported questionnaires that are time-consuming and prone to patient bias. This project implements a supervised learning pipeline to process 18 behavioral, psychological, and demographic features to classify addiction risk while prioritizing clinical safety (100% Recall to eliminate false negatives).

🛠️ Technical Architecture & Pipeline
Data Ingestion & Preprocessing:

Handled missing data via SimpleImputer (mean for numerical, most frequent for categorical).

Normalized numerical features using StandardScaler.

Encoded categorical variables using OneHotEncoder via a unified ColumnTransformer pipeline.

Model Training & Tuning:

Trained and evaluated three candidate algorithms: Logistic Regression (LR), Support Vector Machine (SVM), and Random Forest (RF).

Performed hyperparameter tuning using GridSearchCV with 5-fold cross-validation, optimizing specifically for Recall.

Model Selection & Interpretability:

Selected Logistic Regression as the final production model due to its optimal balance of predictive performance and "White-Box" interpretability.

Extracted model coefficients to perform Feature Importance Analysis (identifying factors like Screen Time Before Bed and Time on Gaming as primary risk drivers).

💻 Tech Stack & Libraries
Programming Language: Python 3.8+

Data Handling: Pandas, NumPy

Machine Learning: Scikit-Learn (Pipelines, ColumnTransformer, GridSearchCV, SVM, Random Forest, Logistic Regression)

Visualization: Matplotlib, Seaborn

📊 Results Summary
Safety Optimization: Both Logistic Regression and Random Forest achieved 100% Recall on the test set, ensuring zero missed at-risk individuals.

Best Model: Logistic Regression was chosen for clinical deployment, achieving an overall accuracy of ~99% while maintaining total transparency via feature weights.

⚙️ Getting Started / Installation
Clone the repository:

Bash
git clone https://github.com/your-username/smartphone-addiction-classifier.git
cd smartphone-addiction-classifier
Install required dependencies:

Bash
pip install pandas numpy scikit-learn matplotlib seaborn
Run the pipeline:
Ensure your dataset (teen_phone_addiction_dataset.csv) is placed in the project directory, then run the Python script or Jupyter Notebook.

