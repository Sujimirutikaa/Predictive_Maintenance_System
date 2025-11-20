import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PredictiveMaintenanceTrainer:
    def __init__(self, data_path='data/predictive_maintenance.csv'):
        self.data_path = data_path
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Display basic info
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Handle missing values if any
        self.df = self.df.dropna()
        
        # Encode categorical variables
        if 'Type' in self.df.columns:
            self.df['Type_encoded'] = self.label_encoder.fit_transform(self.df['Type'])
        
        # Prepare features and target
        feature_columns = [
            'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
            'Torque [Nm]', 'Tool wear [min]'
        ]
        
        if 'Type_encoded' in self.df.columns:
            feature_columns.append('Type_encoded')
            
        self.X = self.df[feature_columns]
        self.y = self.df['Target']
        
        # Also prepare failure type for detailed analysis
        if 'Failure Type' in self.df.columns:
            self.failure_types = self.df['Failure Type']
        
        print(f"Features shape: {self.X.shape}")
        print(f"Target distribution:\n{self.y.value_counts()}")
        
        return self.X, self.y
    
    def split_and_scale_data(self):
        """Split data and scale features"""
        print("Splitting and scaling data...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
    
    def train_models(self):
        """Train multiple ML models"""
        print("Training multiple ML models...")
        
        # Define models
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            #'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM': SVC(random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation score
            if name in ['SVM', 'Logistic Regression']:
                cv_score = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5).mean()
            else:
                cv_score = cross_val_score(model, self.X_train, self.y_train, cv=5).mean()
            
            self.model_scores[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_score': cv_score
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    def select_best_model(self):
        """Select the best performing model"""
        print("\nSelecting best model...")
        
        # Find best model based on F1 score
        best_score = 0
        for name, scores in self.model_scores.items():
            if scores['f1_score'] > best_score:
                best_score = scores['f1_score']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        print(f"Best model: {self.best_model_name} with F1 score: {best_score:.4f}")
        
        # Print comparison table
        print("\nModel Comparison:")
        print("-" * 80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 80)
        for name, scores in self.model_scores.items():
            marker = " *" if name == self.best_model_name else ""
            print(f"{name:<20} {scores['accuracy']:<10.4f} {scores['precision']:<10.4f} {scores['recall']:<10.4f} {scores['f1_score']:<10.4f}{marker}")
    
    def save_model_and_info(self):
        """Save the best model and its information"""
        print("\nSaving model and information...")
        
        # Create directories if they don't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the best model
        joblib.dump(self.best_model, 'models/best_model.pkl')
        
        # Save the scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        # Save label encoder if used
        if hasattr(self, 'label_encoder'):
            joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        
        # Get feature importance if available
        feature_importance = None
        feature_names = list(self.X.columns)
        
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, self.best_model.feature_importances_.tolist()))
        elif hasattr(self.best_model, 'coef_'):
            # For logistic regression, use absolute coefficients
            feature_importance = dict(zip(feature_names, np.abs(self.best_model.coef_[0]).tolist()))
        
        # Prepare model information
        model_info = {
            'model_name': self.best_model_name,
            'accuracy': self.model_scores[self.best_model_name]['accuracy'],
            'precision': self.model_scores[self.best_model_name]['precision'],
            'recall': self.model_scores[self.best_model_name]['recall'],
            'f1_score': self.model_scores[self.best_model_name]['f1_score'],
            'cv_score': self.model_scores[self.best_model_name]['cv_score'],
            'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_names': feature_names,
            'feature_importance': feature_importance,
            'all_model_scores': self.model_scores,
            'dataset_info': {
                'total_samples': len(self.df),
                'features': len(feature_names),
                'target_distribution': self.y.value_counts().to_dict()
            }
        }
        
        # Save model information
        with open('models/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("Model and information saved successfully!")
        print(f"Best model: {self.best_model_name}")
        print(f"Model accuracy: {model_info['accuracy']:.4f}")
        
        return model_info

def main():
    """Main training function"""
    print("=== Predictive Maintenance Model Training ===")
    
    # Initialize trainer
    trainer = PredictiveMaintenanceTrainer()
    
    try:
        # Load and preprocess data
        X, y = trainer.load_and_preprocess_data()
        
        # Split and scale data
        trainer.split_and_scale_data()
        
        # Train models
        trainer.train_models()
        
        # Select best model
        trainer.select_best_model()
        
        # Save model and info
        model_info = trainer.save_model_and_info()
        
        print("\n=== Training Complete ===")
        print("You can now run the web application with: python app.py")
        
    except FileNotFoundError:
        print("Error: Could not find the dataset file.")
        print("Please ensure 'predictive_maintenance.csv' is in the 'data/' folder.")
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main()