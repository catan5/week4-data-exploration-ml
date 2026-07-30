# Week 4 Hands-On Assignment: Data Exploration, Feature Engineering, and GitHub Workflow

**Course: Introduction to Machine Learning, AAI201 32782
**Instructor: 
**Student: Katintha Mwanza  

## Assignment Overview

This assignment focuses on practical data exploration, feature engineering, and basic modeling using Python and scikit-learn. You will also practice using GitHub to document and share your workflow.

## Instructions

1. Set up your GitHub repository  
   - Name: `week4-data-exploration-ml`  
   - Include this README and your Jupyter Notebook.

2. Data Exploration & Feature Engineering  
   - Choose a dataset (e.g., Adult Income, Boston Housing, or another from scikit-learn).  
   - Explore the data, visualize features, and perform feature engineering (encoding, new features).

3. Model Training & Evaluation  
   - Train a simple model (e.g., LogisticRegression or RandomForestClassifier).  
   - Evaluate and interpret your results.

4. Reflection  
   - Briefly reflect on your process and findings in a Markdown cell or separate file.

## How to Run

1. Clone this repository or download the notebook.  
2. Install requirements:  
   pip install numpy pandas matplotlib scikit-learn
3. Open the notebook in Jupyter Notebook or try out the following in Colab:

## Submission

- Push your completed notebook and reflection to this repository.  
- Submit the repository link in Canvas.
Starter Jupyter Notebook (Markdown Format)
# Week 4 Hands-On: Data Exploration, Feature Engineering, and Modeling

## 1. Introduction & Setup

In this assignment, you will practice data exploration, feature engineering, and basic modeling using Python and scikit-learn. You will also document your workflow using GitHub.

Instructions:  
- Complete each section below.  
- Add Markdown cells to explain your steps and findings.  
- Commit your work to your GitHub repository.

---

## 2. Data Loading & Exploration

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# Load a dataset (you may choose another, e.g., Adult Income)
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston.target

# Display first few rows
df.head()

# Summary statistics
df.describe()

# Check for missing values
df.isnull().sum()

# Example: Histogram
df['AGE'].hist(bins=20)
plt.title('Distribution of AGE')
plt.xlabel('AGE')
plt.ylabel('Frequency')
plt.show()

# Example: Scatter plot
plt.scatter(df['RM'], df['target'])
plt.xlabel('Average number of rooms (RM)')
plt.ylabel('House Price (target)')
plt.title('Rooms vs. Price')
plt.show()

# Example: If using a dataset with categorical variables, use pd.get_dummies()
# df = pd.get_dummies(df, columns=['categorical_column'])

# Example: Create a binary feature for "high number of rooms"
df['HIGH_RM'] = (df['RM'] > 6).astype(int)
df[['RM', 'HIGH_RM']].head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))```
