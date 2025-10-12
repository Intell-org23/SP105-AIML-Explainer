from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
model = None
X_train = None
X_test = None
y_train = None
y_test = None
feature_names = None
dataset_info = None
label_encoder = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_csv(df):
    """Validate uploaded CSV file"""
    errors = []
    
    # Check if dataframe is empty
    if df.empty:
        errors.append("CSV file is empty")
        return False, errors
    
    # Check minimum rows
    if len(df) < 10:
        errors.append("Dataset must have at least 10 rows")
    
    # Check for at least 2 columns (features + target)
    if len(df.columns) < 2:
        errors.append("Dataset must have at least 2 columns")
    
    # Check for missing column names
    if df.columns.isnull().any():
        errors.append("All columns must have names")
    
    return len(errors) == 0, errors

def preprocess_data(df, target_column):
    """Preprocess the uploaded dataset"""
    global label_encoder
    
    # Separate features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical features in X
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle categorical target
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    else:
        label_encoder = None
    
    # Handle missing values
    X = X.fillna(X.mean(numeric_only=True))
    
    return X, y

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    """Handle CSV file upload"""
    global dataset_info
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Only CSV files are allowed'}), 400
        
        # Read CSV file
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error reading CSV: {str(e)}'}), 400
        
        # Validate CSV
        is_valid, errors = validate_csv(df)
        if not is_valid:
            return jsonify({'success': False, 'error': '; '.join(errors)}), 400
        
        # Save file info
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save dataset info
        dataset_info = {
            'filename': filename,
            'filepath': filepath,
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'sample_data': df.head(5).to_dict('records')
        }
        
        # Save file to disk
        file.seek(0)  # Reset file pointer
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'message': 'Dataset uploaded successfully',
            'dataset_info': dataset_info
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'}), 500

@app.route('/train_custom_model', methods=['POST'])
def train_custom_model():
    """Train model on uploaded dataset"""
    global model, X_train, X_test, y_train, y_test, feature_names, dataset_info
    
    if dataset_info is None:
        return jsonify({'success': False, 'error': 'No dataset uploaded'}), 400
    
    try:
        # Get target column from request
        data = request.get_json()
        target_column = data.get('target_column')
        
        if not target_column:
            return jsonify({'success': False, 'error': 'Target column not specified'}), 400
        
        # Load dataset
        df = pd.read_csv(dataset_info['filepath'])
        
        # Preprocess data
        X, y = preprocess_data(df, target_column)
        feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        accuracy = model.score(X_test, y_test)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully on custom dataset',
            'accuracy': float(accuracy),
            'n_samples': len(X_train),
            'n_features': len(feature_names),
            'features': feature_names,
            'n_classes': len(np.unique(y))
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train model on sample Iris dataset (original functionality)"""
    global model, X_train, X_test, y_train, y_test, feature_names
    
    try:
        from sklearn.datasets import load_iris
        
        # Load Iris dataset
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
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate_shap', methods=['POST'])
def generate_shap():
    """Generate SHAP explanations"""
    global model, X_train, X_test, feature_names
    
    if model is None:
        return jsonify({'success': False, 'error': 'No model trained'}), 400
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values_for_plot = shap_values[0]
        else:
            shap_values_for_plot = shap_values
        
        # Generate summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_for_plot, X_test, feature_names=feature_names, show=False)
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values_for_plot).mean(axis=0)
        importance_data = [
            {'feature': feature_names[i], 'importance': float(feature_importance[i])}
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
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/explain_instance', methods=['POST'])
def explain_instance():
    """Generate SHAP explanation for single instance"""
    global model, X_test, feature_names
    
    if model is None:
        return jsonify({'success': False, 'error': 'No model trained'}), 400
    
    try:
        data = request.get_json()
        instance_idx = int(data.get('instance_idx', 0))
        
        if instance_idx >= len(X_test):
            return jsonify({
                'success': False,
                'error': f'Instance index out of range. Max: {len(X_test)-1}'
            }), 400
        
        instance = X_test.iloc[instance_idx:instance_idx+1]
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(instance)
        
        # Handle multi-class
        if isinstance(shap_values, list):
            shap_values_single = shap_values[0][0]
            expected_value = explainer.expected_value[0]
        else:
            shap_values_single = shap_values[0]
            expected_value = explainer.expected_value
        
        # Generate force plot
        plt.figure(figsize=(12, 4))
        shap.force_plot(
            expected_value,
            shap_values_single,
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
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate_lime', methods=['POST'])
def generate_lime():
    """Generate LIME explanations for the trained model"""
    global model, X_train, X_test, feature_names
    
    if model is None:
        return jsonify({'success': False, 'error': 'No model trained'}), 400
    
    try:
        from lime.lime_tabular import LimeTabularExplainer
        
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=[str(i) for i in range(len(np.unique(y_train)))],
            mode='classification',
            random_state=42
        )
        
        # Get a few instances to explain
        num_instances = min(5, len(X_test))
        lime_explanations = []
        
        for i in range(num_instances):
            instance = X_test.iloc[i].values
            exp = explainer.explain_instance(
                instance,
                model.predict_proba,
                num_features=len(feature_names)
            )
            
            # Get feature importance from LIME
            lime_weights = exp.as_list()
            lime_explanations.append({
                'instance_idx': i,
                'prediction': int(model.predict([instance])[0]),
                'features': [{'feature': feat, 'weight': float(weight)} for feat, weight in lime_weights]
            })
        
        # Calculate average feature importance across instances
        feature_importance = {}
        for feat in feature_names:
            feature_importance[feat] = 0
        
        for exp in lime_explanations:
            for item in exp['features']:
                # Extract feature name from LIME format (removes comparison operators)
                feat_name = item['feature'].split()[0]
                if feat_name in feature_importance:
                    feature_importance[feat_name] += abs(item['weight'])
        
        # Normalize and sort
        total = sum(feature_importance.values())
        if total > 0:
            feature_importance = {k: v/total for k, v in feature_importance.items()}
        
        importance_data = [
            {'feature': k, 'importance': float(v)}
            for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return jsonify({
            'success': True,
            'feature_importance': importance_data,
            'sample_explanations': lime_explanations[:3],
            'n_instances_explained': num_instances
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/generate_waterfall', methods=['POST'])
def generate_waterfall():
    """Generate SHAP waterfall plot for single instance"""
    global model, X_test, feature_names
    
    if model is None:
        return jsonify({'success': False, 'error': 'No model trained'}), 400
    
    try:
        data = request.get_json()
        instance_idx = int(data.get('instance_idx', 0))
        
        if instance_idx >= len(X_test):
            return jsonify({
                'success': False,
                'error': f'Instance index out of range. Max: {len(X_test)-1}'
            }), 400
        
        instance = X_test.iloc[instance_idx:instance_idx+1]
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(instance)
        
        # Generate waterfall plot
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
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
            'instance_idx': instance_idx
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/explain_instance_lime', methods=['POST'])
def explain_instance_lime():
    """Generate LIME explanation for a single instance"""
    global model, X_train, X_test, y_train, feature_names
    
    if model is None:
        return jsonify({'success': False, 'error': 'No model trained'}), 400
    
    try:
        from lime.lime_tabular import LimeTabularExplainer
        
        data = request.get_json()
        instance_idx = int(data.get('instance_idx', 0))
        
        if instance_idx >= len(X_test):
            return jsonify({
                'success': False,
                'error': f'Instance index out of range. Max: {len(X_test)-1}'
            }), 400
        
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=[str(i) for i in range(len(np.unique(y_train)))],
            mode='classification',
            random_state=42
        )
        
        # Get instance and explain
        instance = X_test.iloc[instance_idx].values
        exp = explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=len(feature_names)
        )
        
        # Get prediction
        prediction = int(model.predict([instance])[0])
        prediction_proba = model.predict_proba([instance])[0]
        
        # Get LIME weights
        lime_weights = exp.as_list()
        explanation_data = [
            {'feature': feat, 'weight': float(weight)}
            for feat, weight in lime_weights
        ]
        
        # Create visualization
        fig = exp.as_pyplot_figure()
        fig.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'plot_image': img_str,
            'prediction': prediction,
            'prediction_proba': prediction_proba.tolist(),
            'explanation': explanation_data,
            'instance_data': X_test.iloc[instance_idx].to_dict()
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/download_results', methods=['POST'])
def download_results():
    """Generate and return analysis report"""
    global model, X_train, X_test, y_train, y_test, feature_names, dataset_info
    
    if model is None:
        return jsonify({'success': False, 'error': 'No model trained'}), 400
    
    try:
        from datetime import datetime
        
        # Generate report content
        report = f"""SP-105 AIML EXPLAINER - ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================

MODEL INFORMATION:
- Model Type: Random Forest Classifier
- Training Samples: {len(X_train)}
- Test Samples: {len(X_test)}
- Number of Features: {len(feature_names)}
- Model Accuracy: {model.score(X_test, y_test):.4f}

FEATURES:
{', '.join(feature_names)}

DATASET INFORMATION:
"""
        if dataset_info:
            report += f"- Filename: {dataset_info.get('filename', 'N/A')}\n"
            report += f"- Total Rows: {dataset_info.get('n_rows', 'N/A')}\n"
            report += f"- Total Columns: {dataset_info.get('n_columns', 'N/A')}\n"
        else:
            report += "- Sample Dataset: Iris\n"
        
        report += "\n================================================\n"
        report += "This report was generated by SP-105 AIML Explainer\n"
        report += "Kennesaw State University - Fall 2025\n"
        
        filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        return jsonify({
            'success': True,
            'report': report,
            'filename': filename
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True, port=5000)