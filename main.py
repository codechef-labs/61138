import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import re

def load_and_preprocess_data(train_path, test_path=None):
    """
    Load and preprocess the Titanic dataset with advanced feature engineering
    """
    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path) if test_path else None
    
    def extract_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        if title_search:
            title = title_search.group(1)
            if title in ['Mlle', 'Ms']:
                return 'Miss'
            elif title in ['Mme']:
                return 'Mrs'
            elif title in ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
                return 'Rare'
            else:
                return title
        return 'Unknown'

    def preprocess(df):
        # Create a copy of the dataframe
        data = df.copy()
        
        # Extract titles from names
        data['Title'] = data['Name'].apply(extract_title)
        
        # Create family size feature
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
        
        # Fill missing ages based on Title and Pclass
        age_by_title_class = data.groupby(['Title', 'Pclass'])['Age'].transform('median')
        data['Age'] = data['Age'].fillna(age_by_title_class)
        
        # Fill remaining missing ages with median
        data['Age'] = data['Age'].fillna(data['Age'].median())
        
        # Create age bands with manual bins to avoid duplicates
        data['AgeBand'] = pd.cut(data['Age'], 
                                bins=[0, 16, 32, 48, 64, np.inf],
                                labels=['Child', 'Young', 'Middle', 'Senior', 'Elderly'])
        
        # Fill missing embarked with mode
        data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
        
        # Fill missing fare with median by Pclass
        fare_by_class = data.groupby('Pclass')['Fare'].transform('median')
        data['Fare'] = data['Fare'].fillna(fare_by_class)
        
        # Create fare bands with manual bins
        data['FareBand'] = pd.qcut(data['Fare'], 
                                  q=5, 
                                  labels=['Lowest', 'Low', 'Medium', 'High', 'Highest'],
                                  duplicates='drop')
        
        # Extract deck from cabin
        data['Deck'] = data['Cabin'].str.extract('([A-Z])', expand=False)
        data['Deck'] = data['Deck'].fillna('Unknown')
        
        # Create categorical features
        data['CabinType'] = np.where(data['Cabin'].isnull(), 'No Cabin', 'Has Cabin')
        
        # Create interaction features
        data['Age*Class'] = data['Age'] * data['Pclass']
        data['Fare*Class'] = data['Fare'] * data['Pclass']
        
        # Drop unnecessary columns
        columns_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
        data = data.drop(columns_to_drop, axis=1)
        
        return data

    # Preprocess both train and test data
    processed_train = preprocess(train_data)
    processed_test = preprocess(test_data) if test_data is not None else None
    
    return processed_train, processed_test

def encode_features(train_df, test_df=None):
    """
    Encode categorical features and scale numerical features
    """
    # Separate categorical and numerical columns
    categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeBand', 'FareBand', 'Deck', 'CabinType']
    numerical_cols = ['Age', 'Fare', 'FamilySize', 'Pclass', 'Age*Class', 'Fare*Class']
    
    # Initialize encoders and scalers
    label_encoders = {}
    scaler = StandardScaler()
    
    # Process train data
    train_encoded = train_df.copy()
    
    # Encode categorical features
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        train_encoded[col] = label_encoders[col].fit_transform(train_df[col])
    
    # Scale numerical features
    train_encoded[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    
    # Process test data if provided
    test_encoded = None
    if test_df is not None:
        test_encoded = test_df.copy()
        # Encode categorical features
        for col in categorical_cols:
            test_encoded[col] = label_encoders[col].transform(test_df[col])
        # Scale numerical features
        test_encoded[numerical_cols] = scaler.transform(test_df[numerical_cols])
    
    return train_encoded, test_encoded

def train_and_evaluate_model(X, y):
    """
    Train a Random Forest model and evaluate using cross-validation
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train the final model on full training data
    model.fit(X, y)
    return model

def create_submission(model, test_encoded, test_ids, output_path):
    """
    Create submission file with predictions
    """
    predictions = model.predict(test_encoded)
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission file created: {output_path}")
    return submission

def main():
    # Load and preprocess data
    train_data, test_data = load_and_preprocess_data('train.csv', 'test.csv')
    
    # Separate features and target for training data
    X_train = train_data.drop('Survived', axis=1)
    y_train = train_data['Survived']
    
    # Encode features
    X_train_encoded, test_data_encoded = encode_features(X_train, test_data)
    
    # Train and evaluate model
    model = train_and_evaluate_model(X_train_encoded, y_train)
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': X_train_encoded.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Create submission file
    if test_data_encoded is not None:
        test_ids = pd.read_csv('test.csv')['PassengerId']
        submission = create_submission(model, test_data_encoded, test_ids, 'submission.csv')

if __name__ == "__main__":
    main()
