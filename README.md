# SP-105 AIML Explainer - Visualizing Black-Box Models

**Course:** 4850 Senior Project  
**Semester:** Fall 2025  
**Team:** SP-105 Visualizing Black-Box Models ALL  

## Project Overview
A web-based application that makes machine learning model predictions interpretable and explainable using SHAP and LIME techniques.

# Clone repository
git clone https://github.com/Intell-org23/SP105-AIML-Explainer.git
cd SP105-AIML-Explainer

# Create virtual environment
python -m venv venv

## Current Status

### ✅ Completed (Weeks 1-2)
- Working Flask application with SHAP integration
- Random Forest model training on Iris dataset
- Global SHAP explanations with summary plots
- Local SHAP explanations with force plots
- Interactive web interface with real-time visualization

[Proof of concept](<Files/Proof of concept 1.pdf>)



### ✅ Completed (Weeks 3-4)
- CSV file upload functionality (up to 100MB)
- Automatic data preprocessing for mixed data types
- Categorical feature encoding (LabelEncoder)
- Target column selection dropdown
- File validation and error handling
- Support for custom datasets with real-world data

[Data set trial result](<Files/trial with data.pdf>)


### ✅ Completed
- LIME integration
- Model upload capability
- Result download capability

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env

# Run development server
python src/app.py

# SP-105 AIML Explainer - Visualizing Black-Box Models

**Course:** 4850 Senior Project  
**Semester:** Fall 2025  
**Team:** SP-105 Visualizing Black-Box Models ALL  

## Project Overview
A web-based application that makes machine learning model predictions interpretable and explainable using SHAP and LIME techniques.

## Team Members
- **Franck Tayo** - Lead Developer (ftayogou@students.kennesaw.edu) 
- **Enzo Nkouekam** - Team Leader, Documentation (ynkoueka@kennesaw.edu)
- **Gloria Kouam** - Developer (gkouamfa@students.kennesaw.edu)
- **Tex Yonzo** - Documentation (tyonzoya@students.kennesaw.edu)

**Supervisor:** Prof. Sharon Perry (sperry46@kennesaw.edu)

## Technology Stack
- **Backend:** Python, Flask, PostgreSQL
- **Frontend:** HTML5, CSS3, JavaScript, Bootstrap
- **ML Libraries:** scikit-learn, SHAP, LIME, XGBoost
- **Visualization:** D3.js, Plotly.js

## Quick Start

### Prerequisites
- Python 3.8 or higher
- PostgreSQL 12+
- Git

### Installation
```bash
# Clone repository
git clone https://github.com/Intell-org23/SP105-AIML-Explainer.git
cd SP105-AIML-Explainer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env

# Run development server
python src/app.py
Access the application at http://localhost:5000
Project Structure
SP105-AIML-Explainer/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── docs/                     # Project documentation
├── src/                      # Source code
│   ├── app.py               # Main Flask application
│   ├── models/              # Database models
│   ├── routes/              # API endpoints
│   ├── services/            # Business logic
│   ├── utils/               # Utility functions
│   └── templates/           # HTML templates
├── static/                  # Static assets (CSS, JS, images)
├── tests/                   # Unit and integration tests
├── data/                    # Sample datasets and temp files
├── scripts/                 # Setup and utility scripts
└── deployment/              # Deployment configurations
Features

-->Interactive ML model explanations using SHAP and LIME
- 🔒 Secure user authentication and session management
-->CSV dataset upload and processing
-->Support for multiple ML models (Random Forest, XGBoost, custom models)
-->Export capabilities (PDF, PNG, JSON)
-->Shareable explanation links

Development Workflow
Branch Naming Convention

feature/feature-name - New features
bugfix/bug-description - Bug fixes
docs/documentation-update - Documentation updates
hotfix/critical-fix - Critical production fixes

Commit Messages
type(scope): description

Types: feat, fix, docs, style, refactor, test, chore
Examples:
feat(data-processor): add CSV validation
fix(explanation-engine): resolve SHAP memory leak
docs(readme): update installation instructions
Contributing

Create feature branch from main
Make changes and test locally
Create pull request with description
Code review by team members
Merge after approval

License
This project is licensed under the MIT License - see the LICENSE file for details.
Documentation

Project Plan
Requirements (SRS)
Design (SDD)
Development Documentation

