from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
import os

app = Flask(__name__)

class PredictiveMaintenancePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.model_info = None
        self.load_model_and_info()
    
    def load_model_and_info(self):
        """Load the trained model and its information"""
        try:
            # Load model
            self.model = joblib.load('models/best_model.pkl')
            
            # Load scaler
            self.scaler = joblib.load('models/scaler.pkl')
            
            # Load label encoder if exists
            try:
                self.label_encoder = joblib.load('models/label_encoder.pkl')
            except:
                self.label_encoder = None
            
            # Load model info
            with open('models/model_info.json', 'r') as f:
                self.model_info = json.load(f)
                
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Please run model_trainer.py first to train the model.")
    
    def predict(self, input_data):
        """Make prediction on input data"""
        if self.model is None:
            return None, "Model not loaded"
        
        try:
            # Prepare input data
            feature_names = self.model_info['feature_names']
            input_array = np.array([[input_data[feature] for feature in feature_names]])
            
            # Scale the input if needed
            model_name = self.model_info['model_name']
            if model_name in ['SVM', 'Logistic Regression']:
                input_array = self.scaler.transform(input_array)
            
            # Make prediction
            prediction = self.model.predict(input_array)[0]
            prediction_proba = self.model.predict_proba(input_array)[0]
            
            # Get confidence
            confidence = max(prediction_proba) * 100
            
            return {
                'prediction': int(prediction),
                'confidence': confidence,
                'failure_probability': prediction_proba[1] * 100 if len(prediction_proba) > 1 else 0
            }, None
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    def get_recommendations(self, prediction_result, input_data):
        """Get detailed recommendations based on prediction"""
        if prediction_result['prediction'] == 0:
            # No failure predicted
            recommendations = {
                'status': 'Normal Operation',
                'risk_level': 'Low',
                'color': 'success',
                'icon': '✅',
                'actions': [
                    'Continue regular maintenance schedule',
                    'Monitor tool wear progression',
                    'Check temperature readings periodically',
                    'Maintain current operational parameters'
                ]
            }
        else:
            # Failure predicted
            failure_prob = prediction_result['failure_probability']
            
            if failure_prob > 80:
                risk_level = 'Critical'
                color = 'danger'
                icon = '🚨'
            elif failure_prob > 60:
                risk_level = 'High'
                color = 'warning'
                icon = '⚠️'
            else:
                risk_level = 'Medium'
                color = 'warning'
                icon = '⚡'
            
            recommendations = {
                'status': 'Maintenance Required',
                'risk_level': risk_level,
                'color': color,
                'icon': icon,
                'actions': self.get_specific_recommendations(input_data)
            }
        
        return recommendations
    
    def get_specific_recommendations(self, input_data):
        """Get specific recommendations based on input parameters"""
        actions = []
        
        # Temperature-based recommendations
        air_temp = input_data.get('Air temperature [K]', 0)
        process_temp = input_data.get('Process temperature [K]', 0)
        
        if air_temp > 305:
            actions.append('Check cooling system - air temperature is elevated')
        if process_temp > 315:
            actions.append('Reduce process temperature - overheating detected')
        
        # Speed-based recommendations
        speed = input_data.get('Rotational speed [rpm]', 0)
        if speed > 2000:
            actions.append('Reduce rotational speed to prevent mechanical stress')
        elif speed < 1200:
            actions.append('Check for mechanical issues - speed is below optimal range')
        
        # Torque-based recommendations
        torque = input_data.get('Torque [Nm]', 0)
        if torque > 50:
            actions.append('High torque detected - check for mechanical binding or tool wear')
        elif torque < 20:
            actions.append('Low torque may indicate tool degradation or power issues')
        
        # Tool wear recommendations
        tool_wear = input_data.get('Tool wear [min]', 0)
        if tool_wear > 200:
            actions.append('🔧 URGENT: Schedule immediate tool replacement')
        elif tool_wear > 150:
            actions.append('Plan tool replacement within next maintenance window')
        elif tool_wear > 100:
            actions.append('Monitor tool condition closely')
        
        # General recommendations
        actions.extend([
            'Perform comprehensive machine inspection',
            'Check all sensors and calibration',
            'Review maintenance logs for patterns',
            'Schedule downtime for preventive maintenance'
        ])
        
        return actions
    
    def get_feature_importance_chart(self):
        """Generate feature importance chart"""
        if not self.model_info or not self.model_info.get('feature_importance'):
            return None
        
        feature_importance = self.model_info['feature_importance']
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        features = [item[0] for item in sorted_features]
        importance = [item[1] for item in sorted_features]
        
        # Create plotly chart
        fig = go.Figure(data=[
            go.Bar(
                x=importance,
                y=features,
                orientation='h',
                marker=dict(
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3'][:len(features)],
                    line=dict(color='rgba(58, 71, 80, 1.0)', width=1)
                )
            )
        ])
        
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# Initialize predictor
predictor = PredictiveMaintenancePredictor()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/model_info')
def get_model_info():
    """Get model information"""
    if predictor.model_info:
        return jsonify(predictor.model_info)
    else:
        return jsonify({'error': 'Model not loaded'}), 500

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """Make prediction based on input data"""
    try:
        input_data = request.json
        
        # Validate input data
        required_features = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ]
        
        for feature in required_features:
            if feature not in input_data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
        
        # Add Type_encoded if model was trained with it
        if predictor.model_info and 'Type_encoded' in predictor.model_info['feature_names']:
            # Default to medium type (encoded as 1) if not provided
            type_mapping = {'L': 0, 'M': 1, 'H': 2}
            machine_type = input_data.get('Type', 'M')
            input_data['Type_encoded'] = type_mapping.get(machine_type, 1)
        
        # Make prediction
        prediction_result, error = predictor.predict(input_data)
        
        if error:
            return jsonify({'error': error}), 500
        
        # Get recommendations
        recommendations = predictor.get_recommendations(prediction_result, input_data)
        
        response = {
            'prediction': prediction_result,
            'recommendations': recommendations
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature_importance')
def get_feature_importance():
    """Get feature importance chart"""
    chart_json = predictor.get_feature_importance_chart()
    if chart_json:
        return jsonify({'chart': chart_json})
    else:
        return jsonify({'error': 'Feature importance not available'}), 404

if __name__ == '__main__':
    print("Starting Predictive Maintenance System...")
    if predictor.model_info:
        print(f"Model loaded: {predictor.model_info['model_name']}")
        print(f"Accuracy: {predictor.model_info['accuracy']:.4f}")
    else:
        print("Warning: Model not loaded. Please run model_trainer.py first.")
    
    app.run(debug=True)