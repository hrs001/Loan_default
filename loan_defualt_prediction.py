import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier

# Loading dataset
dataset = pd.read_csv("/Users/harshsrivastava/Downloads/credit-default-prediction-ai-big-data/train.csv")

# Dropping Id
dataset = dataset.drop(columns=['Id'])
# Splitting the data
X = dataset.drop(columns=['Credit Default']).copy()
y = dataset['Credit Default'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,       # 20% test set
    random_state=42,     # Reproducibility
    stratify=y           # Preserve class balance
)


# Preprocessing dataset
Home_Ownership = dataset['Home Ownership'].unique()
Years_in_current_job = dataset['Years in current job'].unique()
purpose = dataset['Purpose'].unique()
preprocessor = ColumnTransformer(
    transformers=[
        ("Annual Income", SimpleImputer(strategy="mean"), ["Annual Income"]),
        ("Months since last delinquent", SimpleImputer(strategy="most_frequent"), ["Months since last delinquent"]), 
        ("Bankruptcies", SimpleImputer(strategy="most_frequent"), ["Bankruptcies"]),                       
        ("Credit Score", SimpleImputer(strategy="mean"), ["Credit Score"]), 

        ("Term", OrdinalEncoder(categories=[['Short Term', 'Long Term']]), ["Term"]), 
        ("Purpose", OrdinalEncoder(categories=[purpose]), ["Purpose"]), 
        ("Home Ownership", OrdinalEncoder(categories=[Home_Ownership]), ["Home Ownership"]), 
    ]
)

# Preprocessing on a single columns
num_imputer = SimpleImputer(strategy="most_frequent")
X_train["Years in current job"] = num_imputer.fit_transform(X_train[["Years in current job"]])[:, 0]
X_test["Years in current job"] = num_imputer.transform(X_test[["Years in current job"]])[:, 0]
encoder = OrdinalEncoder()
X_train["Years in current job"] = encoder.fit_transform(X_train[['Years in current job']])
X_test["Years in current job"] = encoder.transform(X_test[['Years in current job']])



#### Data Analysis ####
plt.title("Housing Status")
plt.bar(dataset["Credit Default",] )



# Define pipeline steps
pipeline = Pipeline([
    ('preprocessing_!', preprocessor),
#   ('Scaling_1', StandardScaler()),
])

# Transfroming variables
X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# Training & Predicting
lgb = LGBMClassifier(
    n_estimators=100,    # Number of boosting stages
    learning_rate=0.1,
    max_depth=-1,        # -1 = no limit
    num_leaves=31,       # Max leaves per tree
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42
)

# --- RFE Feature Selection ---
rfe = RFE(lgb, n_features_to_select=12)
rfe.fit(X_train_transformed, y_train)

# Use only best features
X_train_selected = X_train_transformed[:, rfe.support_]
X_test_selected = X_test_transformed[:, rfe.support_]

print("Selected features (RFE):", preprocessor.get_feature_names_out()[rfe.support_])

# Fit on RFE-selected features
rfe.fit(X_train_selected, y_train)

y_pred = rfe.predict(X_test_selected)

# --- Metrics ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))