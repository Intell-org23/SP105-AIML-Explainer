from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import shap
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Global variables to store model and data
model = None
X_train = None
X_test = None
y_train = None
y_test = None
feature_names = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train a simple Random Forest model on Iris dataset"""
    global model, X_train, X_test, y_train, y_test, feature_names
    
    try:
        # Load sample data (Iris dataset)
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target
        feature_names = iris.feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get accuracy
        accuracy = model.score(X_test, y_test)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'accuracy': float(accuracy),
            'n_samples': len(X_train),
            'n_features': len(feature_names)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/generate_shap', methods=['POST'])
def generate_shap():
    """Generate SHAP explanations for the trained model"""
    global model, X_train, X_test, feature_names
    
    if model is None:
        return jsonify({
            'success': False,
            'error': 'No model trained yet. Train a model first.'
        }), 400
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for test set
        shap_values = explainer.shap_values(X_test)
        
        # Generate summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[0], X_test, feature_names=feature_names, show=False)
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        # Get feature importance
        feature_importance = np.abs(shap_values[0]).mean(axis=0)
        importance_data = [
            {
                'feature': feature_names[i],
                'importance': float(feature_importance[i])
            }
            for i in range(len(feature_names))
        ]
        importance_data.sort(key=lambda x: x['importance'], reverse=True)
        
        return jsonify({
            'success': True,
            'plot_image': img_str,
            'feature_importance': importance_data,
            'n_samples_explained': len(X_test)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/explain_instance', methods=['POST'])
def explain_instance():
    """Generate SHAP explanation for a single instance"""
    global model, X_test, feature_names
    
    if model is None:
        return jsonify({
            'success': False,
            'error': 'No model trained yet.'
        }), 400
    
    try:
        # Get instance index (default to first test sample)
        instance_idx = int(request.json.get('instance_idx', 0))
        
        if instance_idx >= len(X_test):
            return jsonify({
                'success': False,
                'error': f'Instance index out of range. Max: {len(X_test)-1}'
            }), 400
        
        # Get the instance
        instance = X_test.iloc[instance_idx:instance_idx+1]
        
        # Create explainer and get SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(instance)
        
        # Generate force plot
        plt.figure(figsize=(12, 4))
        shap.force_plot(
            explainer.expected_value[0],
            shap_values[0][0],
            instance,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        # Get prediction
        prediction = model.predict(instance)[0]
        prediction_proba = model.predict_proba(instance)[0]
        
        return jsonify({
            'success': True,
            'plot_image': img_str,
            'prediction': int(prediction),
            'prediction_proba': prediction_proba.tolist(),
            'instance_data': instance.to_dict('records')[0]
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)