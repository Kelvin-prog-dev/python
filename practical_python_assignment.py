import graphviz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# STEP 1: Raw Data Collection

print(" ------- Exploring the data collected ------- ")

raw_data = {
'Student_ID': [1001, 1002, 1003, 1004, 1005, 1001, 1006, 1007, 1008, 1009],
'County': ['Nakuru', 'Narok', 'Nakuru', 'Narok', 'Kakamega',
'Nakuru', 'Nairobi', None, 'Narok', 'Nakuru'],
'Study_Hours_Per_Week': [15, 8, 20, None, 12, 15, 30, 10, 25, 18],
'Attendance_Percent': [85, 60, 92, 55, 78, 85, 95, 65, 88, None],
'Fee_Balance_Ksh': [0, 15000, 0, 25000, 5000, 0, 0, 18000, 2000, 12000],
'Previous_Grade': [72, 48, 85, 45, 65, 72, 90, 50, 82, 58],
'Student_Type': ['Regular', 'Regular', 'Scholarship', 'Regular',
'Scholarship', 'Regular', 'Regular', 'Scholarship',
'Regular', 'Regular'],
'Academic_Risk': [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
}

# Create dataframe for the data
df = pd.DataFrame(raw_data)

# Gather info about the data
print(f"DataFrame Shape: {df.shape}\n")

print("Data Types in the DataFrame:\n")

print(df.dtypes, "\n")

print(df.describe(), "\n")

print(df.info(), "\n")

# STEP 2: Data Cleaning (Duplicates & Missing Values)

print(" ------- Data rangling -------\n")

# Checking for duplicate values (using duplicated())
duplicates = df.duplicated()
print('Duplicates:\n\n', duplicates)

# Handling duplicates
df_keep_last = df.drop_duplicates(subset='Student_ID', keep='last')
df = df_keep_last
print("Keep last occurrence; no duplicates")
print(df_keep_last)
print(f"Shape: {df_keep_last.shape}\n")


# Return the count of NaNs per column
print("Missing values per column:")
print(df.isna().sum(), '\n')

# Clean: Fill missing values (median for numerical, "Uknown" for categorical)
# Fill the NaN value in the County column

df['County'] = df['County'].fillna("Uknown")
df['Study_Hours_Per_Week'] = df['Study_Hours_Per_Week'].fillna(df['Study_Hours_Per_Week'].median())
df['Attendance_Percent'] = df['Attendance_Percent'].fillna(df['Attendance_Percent'].fillna(df['Attendance_Percent'].median()))


# STEP 3: EDA
sns.set_theme(style="whitegrid")
plt.figure(figsize=(15, 10))

# Academic risk by County (Plot 1)
plt.subplot(2, 2, 1)
sns.barplot(x='County', y='Academic_Risk', data=df,  hue='County', palette='Set2', legend=False)
plt.title("Proportion of Academic Risk by County")
plt.ylabel('Proportion At Risk')

# Fee Balance vs Academic Risk (Plot 2)
plt.subplot(2, 2, 2)
sns.boxplot(x='Academic_Risk', y='Fee_Balance_Ksh', data=df, hue='Academic_Risk', palette='Set2', legend=False)
plt.title("Fee Balance (Ksh) by Academic Risk")
plt.xticks([0, 1], ['Not At Risk (0)', 'At Risk (1)'])

# Previous grade vs Academic Risk (Plot 3)
plt.subplot(2, 2, 3)
sns.boxplot(x='Academic_Risk', y='Previous_Grade', data=df, hue='Academic_Risk', palette='Set2', legend=False)
plt.title("Previous Grade by Academic Risk")
plt.xticks([0, 1], ['Not At Risk (0)', 'At Risk (1)'])

# Attendance Percent vs Academic Risk (Plot 4)
plt.subplot(2, 2, 4)
sns.boxplot(x='Academic_Risk', y='Attendance_Percent', data=df, hue='Academic_Risk', palette='Set2', legend=False)
plt.title("Attendance Percent by Academic Risk")
plt.xticks([0, 1], ['Not At Risk (0)', 'At Risk (1)'])

plt.tight_layout()
plt.show()

# Summary of the statistics
print("\n Mean Values by Academic Risk")
print(df.groupby('Academic_Risk')[['Previous_Grade', 'Study_Hours_Per_Week', 'Attendance_Percent', 'Fee_Balance_Ksh']].mean())

# STEP 4: Feature Engineering
# Feature engineering
bins = [0, 60, 80, 100]
labels = ['Low', 'Medium', 'High']

# Create new features 
df['Attendance_Category'] = pd.cut(df['Attendance_Percent'], bins=bins, labels=labels)

df['Fee_Debt_Indicator'] = np.where(df['Fee_Balance_Ksh'] > 0, 'Has_Debt', 'Clear')

print("\n --- Feature Engineered Data Sample --- ")
print(df[['Attendance_Percent', 'Attendance_Category', 'Fee_Balance_Ksh', 'Fee_Debt_Indicator']].head())

# STEP 5: Encoding and Feature Scaling (Encoded and standardized data)
# Convert categorical variables into dummy/indicator variables (using pd.get_dummies())
# Each variable is converted in as many 0/1 variables as there are different values. 
df_encoded = pd.get_dummies(df, columns=['County', 'Student_Type', 'Attendance_Category', 'Fee_Debt_Indicator'], drop_first=True)

# Choose the target feature (Academic_Risk)
X = df_encoded.drop(columns=['Student_ID', 'Academic_Risk']) # Dropped to prevent data leakage and overfitting
y = df_encoded['Academic_Risk'] # Extracts this column as a pd series for prediction after training

## Train-test split (for large datasets)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print("\n --- Scaled Training Data ---")
print(X_train_scaled.head(3))

# STEP 6: Correlation heatmap
number_cols = ['Study_Hours_Per_Week', 'Attendance_Percent', 'Fee_Balance_Ksh', 'Previous_Grade', 'Academic_Risk']
corr_matrix = df[number_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


# STEP 7: Decision Tree and Graphviz
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
dt_classifier.fit(X_train_scaled, y_train)

# Export to Graphviz
dot_data = export_graphviz(
	dt_classifier,
	out_file=None,
	feature_names=X_train_scaled.columns,
	class_names=['Not At Risk (0)', 'At Risk (1)'],
	filled=True,
	rounded=True,
	special_characters=True
)

graph = graphviz.Source(dot_data)
# display(graph) # display in Jupyter Notebook

# Save the graphic image
graph.render("Academic_Risk_Decision_Tree", format="png", cleanup=True)
print("\n --- Graphviz Visualization ---")
print("Decision Tree image saved as 'Academic_Risk_Decision_Tree.png'")


# Feature interpretation
feature_importances = pd.Series(dt_classifier.feature_importances_, index=X_train_scaled.columns)
root_feature = feature_importances.idxmax()

print("\n --- Decision Tree Interpretation ---")
print(f"Root Decision Feature (Highest Information Gain): '{root_feature}'")
print("\nMain Decision Paths:")
print("1. The tree first splits on the root feature(e.g., Attendance_Percent or Previous_Grade).)")
print("2. If the value is below the learned threshold, it moves to the left child node (higher risk of Academic_Risk = 1).")
print("3. Subsequent splits (up to max_depth=3) check secondary factors like Fee_Balance_Ksh or Study_Hours_Per_Week.")
print("4. Leaf nodes provide the final classification (0 or 1) based on the majority class of samples reaching that node.")