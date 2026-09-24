import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
import joblib
import os

def load_and_preprocess_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # 1. Clean messy column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # 2. Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    # 3. Drop 'loanid'
    if 'loanid' in df.columns:
        df.drop('loanid', axis=1, inplace=True)
        
    # Rename target column for consistency with EDA
    if 'default' in df.columns:
        df.rename(columns={'default': 'loan_status'}, inplace=True)
        
    # 4. Handle Outliers (IQR Method) as in EDA
    print("Handling outliers...")
    outlier_cols = ['income', 'loanamount']
    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        
    return df

def main():
    data_path = 'Loan_default.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = load_and_preprocess_data(data_path)
    
    # Downsample for faster execution of tuning across multiple models
    if len(df) > 50000:
        print("Downsampling data to 50,000 rows for faster training...")
        df = df.sample(n=50000, random_state=42)
    
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    from sklearn.model_selection import RandomizedSearchCV
    import json

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    # Define hyperparameter grids for tuning
    param_grids = {
        "Logistic Regression": {
            'classifier__C': [0.1, 1.0, 10.0]
        },
        "Decision Tree": {
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10]
        },
        "KNN": {
            'classifier__n_neighbors': [3, 5, 7, 9]
        },
        "Random Forest": {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20]
        }
    }

    evaluation_results = {}

    for name, model in models.items():
        print(f"\nTraining and tuning {name}...")
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        # Hyperparameter tuning using RandomizedSearchCV
        search = RandomizedSearchCV(
            clf, 
            param_distributions=param_grids[name], 
            n_iter=5, # Keep it small for speed
            cv=3, 
            scoring='accuracy', 
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        best_clf = search.best_estimator_
        
        print(f"Best params for {name}: {search.best_params_}")
        
        # Predictions
        y_train_pred = best_clf.predict(X_train)
        y_test_pred = best_clf.predict(X_test)
        y_test_proba = best_clf.predict_proba(X_test)[:, 1] if hasattr(best_clf, "predict_proba") else None
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        recall = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        
        roc_auc = None
        fpr = None
        tpr = None
        if y_test_proba is not None:
            roc_auc = roc_auc_score(y_test, y_test_proba)
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Feature Importance (for Random Forest and Decision Tree)
        feature_importances = None
        if name in ["Decision Tree", "Random Forest"]:
            try:
                feature_names = best_clf.named_steps['preprocessor'].get_feature_names_out()
                importances = best_clf.named_steps['classifier'].feature_importances_
                feature_importances = dict(zip(feature_names, importances))
                # Sort them
                feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
            except Exception as e:
                print(f"Could not get feature importances: {e}")

        evaluation_results[name] = {
            "Train Accuracy": float(train_acc),
            "Test Accuracy": float(test_acc),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1 Score": float(f1),
            "ROC AUC": float(roc_auc) if roc_auc else None,
            "Confusion Matrix": cm.tolist(),
            "ROC Curve": {"fpr": fpr.tolist() if fpr is not None else [], "tpr": tpr.tolist() if tpr is not None else []},
            "Feature Importances": feature_importances
        }
        print(f"Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")
        
        # Save model
        model_filename = f'model_{name.replace(" ", "_").lower()}.pkl'
        print(f"Saving model to {model_filename}...")
        joblib.dump(best_clf, model_filename)

    # Save evaluation results
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(evaluation_results, f)
    print("\nSaved evaluation_metrics.json")
    print("Done!")

if __name__ == '__main__':
    main()
