### Final ###
###==================================================###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier


import os
os.makedirs('Visuals/before_train', exist_ok=True)
os.makedirs('Visuals/after_train', exist_ok=True)
os.makedirs('Visuals/before_test', exist_ok=True)
os.makedirs('Visuals/after_test', exist_ok=True)


train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')


def display(df, is_train=True):
    print(f"===Dispaly ({'Train' if is_train else 'Test'})===")
    try:
        print("Dataset Shape:", df.shape)

        print("\nColumns and Data Types:")
        print(df.dtypes)

        print("\nMissing Values:")
        df.replace(r'^\s*\?\ ?\s*$', np.nan, regex=True, inplace=True)
        print(df.isnull().sum())

        print("\nDuplicated: " + str(df.duplicated().sum()))

        print("\nFirst few rows:")
        print(df.head(3))
        print("\n" + "=" * 50 + "\n")
    except Exception as e:
        print(f"Error loading files: {e}")
        print("\n" + "=" * 50 + "\n")


def visualization(when, df):
    print("===The (" + when + ") Visualization===")
    df.columns = df.columns.str.strip()
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace(r'^\s*\?\ ?\s*$', np.nan, regex=True, inplace=True)
    sns.set_theme(style="whitegrid")

    # target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Income', hue='Income', palette='Set2', legend=False)
    plt.title('Income Distribution')
    plt.xlabel('Income')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('Visuals/' + when + '/' + when + '_target_dist.png')
    plt.close()

    # age distribution by (Income)
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='age', hue='Income', multiple="stack", bins=30, palette='Set2')
    plt.title('Age Distribution by Income')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('Visuals/' + when + '/' + when + '_age_dist.png')
    plt.close()

    # correlation matrix
    plt.figure(figsize=(8, 6))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('Visuals/' + when + '/' + when + '_corr_matrix.png')
    plt.close()

    print(f"EDA for the ({when}) data plots generated.")
    print("\n" + "=" * 50 + "\n")


def clean_and_preprocess(df, is_train=True, encoders=None, scaler=None, selected_features=None):
    print(f"===Clean and Preprocess ({'Train' if is_train else 'Test'})===")
    df = df.copy()

    # stripping w replacing ? with NaN
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace(r'^\s*\?\ ?\s*$', np.nan, regex=True, inplace=True)

    # handling missing w duplicates
    df.dropna(inplace=True)
    print(f"\nAfter droping missing values:\n{df.isnull().sum()}")
    df.drop_duplicates(inplace=True)
    print(f"\nAfter droping Duplicated: {df.duplicated().sum()}")

    # encode target
    if 'Income' in df.columns:
        df['Income'] = df['Income'].astype(str).str.replace('.', '', regex=False).str.strip()
        df['Income'] = df['Income'].map({'<=50K': 0, '>50K': 1})

    # one-hot encoding using dummies
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'native-country']
    cat_cols = [col for col in cat_cols if col in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    if is_train:
        selected_features = df.columns
    else:
        df = df.reindex(columns=selected_features, fill_value=0)

    # feature selection using correlation
    if is_train:
        corr = df.corr()['Income'].abs()
        selected_features = corr[corr > 0.1].index.tolist()
        print("\nSelected Features:")
        print(selected_features)

    # bn5ly el-train w el-test yst5dmo nfs el-columns
    features_to_keep = [col for col in selected_features if col in df.columns]
    df = df[features_to_keep]

    # numerical scaling
    base_num_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    num_cols = [col for col in base_num_cols if col in df.columns]
    if len(num_cols) > 0:
        if is_train:
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
        else:
            df[num_cols] = scaler.transform(df[num_cols])

    # balancing with SMOTE
    if is_train and 'Income' in df.columns:
        X = df.drop('Income', axis=1)
        y = df['Income']
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        df = pd.concat([
            pd.DataFrame(X_resampled, columns=X.columns),
            pd.DataFrame(y_resampled, columns=['Income'])
        ], axis=1)
        print("\nAfter SMOTE balancing:")
        print(df['Income'].value_counts())

    print("\nPreprocessed Train Shape:", df.shape)
    print("\nFirst row of processed train data:")
    print(df.iloc[0])
    print("\n" + "=" * 50 + "\n")

    return df, encoders, scaler, selected_features


def logistic_regression_model(train_df, test_df):
    print("=== Training Logistic Regression Model ===")
    X = train_df.drop('Income', axis=1)
    y = train_df['Income']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # model training
        # el-max_iter 34an el-convergence
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # validation evaluation
    y_val_pred = model.predict(X_val)
    print("=== Validation Results ===")
    print("Accuracy:", accuracy_score(y_val, y_val_pred))

    # test evaluation
    X_test = test_df.drop('Income', axis=1)
    y_test = test_df['Income']
    y_test_pred = model.predict(X_test)

    print("\n=== Test Results ===")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))

    # bouns: confusion matrix visualization
    plt.figure(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'], cbar=False)
    plt.title("Logistic Regression - Test Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig('Visuals/lr_confusion_matrix.png')
    plt.close()

    print("\nLR Visualizations saved.")
    print("=" * 50 + "\n")
    return model


def svm_model(train_df, test_df):
    print("=== Training Support Vector Machine (SVM) ===")
    X = train_df.drop('Income', axis=1)
    y = train_df['Income']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # model training
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)

    # validation evaluation
    y_val_pred = svm.predict(X_val)
    print("=== Validation Results ===")
    print("Accuracy:", accuracy_score(y_val, y_val_pred))

    # test evaluation
    X_test = test_df.drop('Income', axis=1)
    y_test = test_df['Income']
    y_test_pred = svm.predict(X_test)

    print("\n=== Test Results ===")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))

    # bouns: confusion matrix visualization
    plt.figure(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'], cbar=False)
    plt.title("SVM - Test Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig('Visuals/svm_confusion_matrix.png')
    plt.close()

    print("\nSVM Visualizations saved.")
    print("=" * 50 + "\n")
    return svm


def decision_tree_model(train_df, test_df):
    print("=== Training Decision Tree Model ===")
    X = train_df.drop('Income', axis=1)
    y = train_df['Income']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # model training
        # el-max_depth=10 34an n3ml stops l el-tree from growing too deep w t3ml memorize l el-data
    dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train)

    # validation evaluation
    y_val_pred = dt_model.predict(X_val)
    print("=== Validation Results ===")
    print("Accuracy:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))

    # test evaluation
    X_test = test_df.drop('Income', axis=1)
    y_test = test_df['Income']

    y_test_pred = dt_model.predict(X_test)
    print("\n=== Test Results ===")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))

    # bouns:
        # confusion matrix visualization
    plt.figure(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=['<=50K', '>50K'],
        yticklabels=['<=50K', '>50K'],
        cbar=False
    )
    plt.title("Test Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig('Visuals/dt_confusion_matrix.png')
    plt.close()

        # decision tree plot
    plt.figure(figsize=(22, 10))
    # 3mlna el-plot max_depth=4 34an el-sora
    plot_tree(
        dt_model,
        feature_names=X.columns,
        class_names=['<=50K', '>50K'],
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=4
    )
    plt.title("Decision Tree", fontsize=16)
    plt.savefig('Visuals/decision_tree_final.png')
    plt.close()

    print("\nVisualizations saved to 'Visuals' folder.")
    print("=" * 50 + "\n")

    return dt_model


def random_forest_model(train_df, test_df):
    print("=== Training Random Forest Model ===")
    X = train_df.drop('Income', axis=1)
    y = train_df['Income']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # model training
        # el-n_estimators=100 hy3ml build l 100 trees.
        # w el-n_jobs=-1 to use all the computer's cpu cores 34an n3ml train asr3
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # validation evaluation
    y_val_pred = rf_model.predict(X_val)
    print("=== Validation Results ===")
    print("Accuracy:", accuracy_score(y_val, y_val_pred))

    # test evaluation
    X_test = test_df.drop('Income', axis=1)
    y_test = test_df['Income']
    y_test_pred = rf_model.predict(X_test)

    print("\n=== Test Results ===")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))

    # bonus:
        # extract el-feature importance
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\nTop 5 Most Important Features:")
    print(feature_importances.head(5).to_string(index=False))

        # confusion matrix visualization
    plt.figure(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'], cbar=False)
    plt.title("Random Forest - Test Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig('Visuals/rf_confusion_matrix.png')
    plt.close()

    print("\n✅ RF Visualizations saved.")
    print("=" * 50 + "\n")
    return rf_model


def tune_and_evaluate_models(train_df, test_df):
    print("=== Hyperparameter Tuning & Final Evaluation ===")

    X_train = train_df.drop('Income', axis=1)
    y_train = train_df['Income']

    X_test = test_df.drop('Income', axis=1)
    y_test = test_df['Income']

    # logistic regression tuning
    print("\nTuning Logistic Regression...")
    log_param_grid = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    log_grid = GridSearchCV(LogisticRegression(max_iter=1000), log_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    log_grid.fit(X_train, y_train)
    print("Best LR Params:", log_grid.best_params_)

    # svm tuning
    print("\nTuning SVM (Warning: This may take several minutes)...")
    svm_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    # cv=3 used instead of 5 to save computational time
    svm_grid = GridSearchCV(SVC(random_state=42), svm_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    svm_grid.fit(X_train, y_train)
    print("Best SVM Params:", svm_grid.best_params_)

    # decision tree tuning
    print("\nTuning Decision Tree...")
    dt_param_grid = {'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 10, 20]}
    dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    dt_grid.fit(X_train, y_train)
    print("Best DT Params:", dt_grid.best_params_)

    # random forest tuning
    print("\nTuning Random Forest...")
    rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print("Best RF Params:", rf_grid.best_params_)

    # final evaluation 3la el-unseen test
    print("\n" + "=" * 50)
    print("FINAL TEST SET ACCURACY (Best Models)")
    print("=" * 50)
    print("Logistic Regression: {:.4f}".format(accuracy_score(y_test, log_grid.predict(X_test))))
    print("SVM:                 {:.4f}".format(accuracy_score(y_test, svm_grid.predict(X_test))))
    print("Decision Tree:       {:.4f}".format(accuracy_score(y_test, dt_grid.predict(X_test))))
    print("Random Forest:       {:.4f}".format(accuracy_score(y_test, rf_grid.predict(X_test))))
    print("=" * 50 + "\n")

    return log_grid.best_estimator_, svm_grid.best_estimator_, dt_grid.best_estimator_, rf_grid.best_estimator_


if __name__ == "__main__":
    display(train_df, is_train=True)
    visualization('before_train', train_df)
    display(test_df, is_train=False)
    visualization('before_test', test_df)

    train_clean, encoders, scaler, selected_features = clean_and_preprocess(train_df, is_train=True)
    test_clean, _, _, _ = clean_and_preprocess(test_df, is_train=False, encoders=encoders, scaler=scaler, selected_features=selected_features)

    display(train_clean, is_train=True)
    visualization('after_train', train_clean)
    display(test_clean, is_train=False)
    visualization('after_test', test_clean)

    train_clean.to_csv('train_data_clean.csv', index=False)
    test_clean.to_csv('test_data_clean.csv', index=False)

    lr_model = logistic_regression_model(train_clean, test_clean)
    trained_svm = svm_model(train_clean, test_clean)
    dt_model = decision_tree_model(train_clean, test_clean)
    rf_model = random_forest_model(train_clean, test_clean)

    best_lr, best_svm, best_dt, best_rf = tune_and_evaluate_models(train_clean, test_clean)
