from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
import requests
import os
from dotenv import load_dotenv
import json
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_API_URL = "https://api.cohere.ai/v1/generate"

def get_llm_explanation(customer_data, churn_prob, churn_prediction):
    try:
        if not COHERE_API_KEY or COHERE_API_KEY == "YOUR_COHERE_API_KEY":
            return generate_fallback_explanation(churn_prob, customer_data)
        
        profile_summary = f"""
Customer Profile:
- Tenure: {customer_data.get('tenure', 'N/A')} months
- Monthly Charges: ${customer_data.get('MonthlyCharges', 'N/A')}
- Total Charges: ${customer_data.get('TotalCharges', 'N/A')}
- Contract: {customer_data.get('Contract', 'N/A')}
- Internet Service: {customer_data.get('InternetService', 'N/A')}
- Online Security: {customer_data.get('OnlineSecurity', 'N/A')}
- Tech Support: {customer_data.get('TechSupport', 'N/A')}
"""
        
        prompt = f"""You are a customer retention expert analyzing churn risk. Based on the customer profile below, provide a concise explanation of why this customer has a {churn_prob:.1f}% churn risk.

{profile_summary}

Churn Prediction: {"HIGH RISK - Customer will likely churn" if churn_prediction == 1 else "LOW RISK - Customer will likely stay"}

Provide a brief response (2-3 sentences) with:
1. Main risk factor
2. One specific retention action

Keep it professional and actionable."""

        headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "command",
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.8,
            "k": 0,
            "p": 0.9,
            "stop_sequences": []
        }
        
        response = requests.post(
            COHERE_API_URL,
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            explanation = result.get('generations', [{}])[0].get('text', '').strip()
            return explanation if explanation else generate_fallback_explanation(churn_prob, customer_data)
        else:
            print(f"Cohere API Error: {response.status_code}")
            return generate_fallback_explanation(churn_prob, customer_data)
            
    except Exception as e:
        print(f"LLM Error: {e}")
        return generate_fallback_explanation(churn_prob, customer_data)

def generate_fallback_explanation(churn_prob, customer_data):
    if churn_prob > 0.7:
        risk_assessment = "🔴 CRITICAL RISK - This customer has a high probability of churning and requires immediate intervention."
        action = "Schedule an urgent retention call within 24-48 hours."
    elif churn_prob > 0.5:
        risk_assessment = "🟠 HIGH RISK - This customer shows significant churn indicators."
        action = "Contact the customer within 3-5 days with a targeted retention offer."
    elif churn_prob > 0.3:
        risk_assessment = "🟡 MEDIUM RISK - Monitor this customer's account activity closely."
        action = "Reach out proactively with service improvements or loyalty incentives."
    else:
        risk_assessment = "🟢 LOW RISK - This customer appears stable and satisfied."
        action = "Focus on upselling or cross-selling opportunities."
    
    tenure = customer_data.get('tenure', 0)
    contract = customer_data.get('Contract', 'Month-to-month')
    
    additional_insight = ""
    if tenure < 6:
        additional_insight = "As a relatively new customer, focus on ensuring excellent onboarding and service quality."
    elif contract == "Month-to-month":
        additional_insight = "The month-to-month contract provides flexibility to leave. Consider offering a discount for a longer commitment."
    
    explanation = f"{risk_assessment} {action} {additional_insight}"
    return explanation.strip()

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

df = pd.read_csv('telco_churn.csv')
print(f"Original data shape: {df.shape}")
print(f"Original columns: {df.columns.tolist()}")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df = df.drop('customerID', axis=1)
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(df[col])
    label_encoders[col] = le

model_features = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else None

print(f"\nModel features after correlation filtering:")
print(model_features)
print(f"Total features: {len(model_features) if model_features else 'Unknown'}")
print("✅ Model loaded successfully\n")

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Telco Churn Prediction API with LLM Explanation', 'status': 'running'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        input_df = pd.DataFrame([data])
        
        print(f"\nInput received with fields: {list(input_df.columns)}")
        
        if 'TotalCharges' in input_df.columns:
            input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
        
        for col in categorical_cols:
            if col in input_df.columns:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except ValueError as e:
                    return jsonify({
                        'error': f'Invalid value for {col}. Valid options: {label_encoders[col].classes_.tolist()}'
                    }), 400
        
        if model_features:
            missing_cols = [col for col in model_features if col not in input_df.columns]
            if missing_cols:
                return jsonify({'error': f'Missing required features: {missing_cols}'}), 400
            
            input_df = input_df[model_features]
        
        input_df = input_df.apply(pd.to_numeric, errors='coerce')
        
        if input_df.isnull().any().any():
            nan_cols = input_df.columns[input_df.isnull().any()].tolist()
            return jsonify({'error': f'Invalid/missing values in: {nan_cols}'}), 400
        
        print(f"Features used for prediction: {list(input_df.columns)}")
        
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        result = {
            'churn': int(prediction),
            'churn_label': 'Yes' if prediction == 1 else 'No',
            'no_churn_probability': float(probability[0]),
            'churn_probability': float(probability[1]),
            'confidence': float(max(probability))
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/predict-with-explanation', methods=['POST'])
def predict_with_explanation():
    try:
        data = request.get_json()
        
        input_df = pd.DataFrame([data])
        
        print(f"\nInput received with fields: {list(input_df.columns)}")
        
        if 'TotalCharges' in input_df.columns:
            input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
        
        for col in categorical_cols:
            if col in input_df.columns:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except ValueError as e:
                    return jsonify({
                        'error': f'Invalid value for {col}. Valid options: {label_encoders[col].classes_.tolist()}'
                    }), 400
        
        if model_features:
            missing_cols = [col for col in model_features if col not in input_df.columns]
            if missing_cols:
                return jsonify({'error': f'Missing required features: {missing_cols}'}), 400
            
            input_df = input_df[model_features]
        
        input_df = input_df.apply(pd.to_numeric, errors='coerce')
        
        if input_df.isnull().any().any():
            nan_cols = input_df.columns[input_df.isnull().any()].tolist()
            return jsonify({'error': f'Invalid/missing values in: {nan_cols}'}), 400
        
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        churn_prob = probability[1] * 100
        
        llm_explanation = get_llm_explanation(data, churn_prob, prediction)
        
        result = {
            'churn': int(prediction),
            'churn_label': 'Yes' if prediction == 1 else 'No',
            'no_churn_probability': float(probability[0]),
            'churn_probability': float(probability[1]),
            'confidence': float(max(probability)),
            'explanation': llm_explanation,
            'risk_level': 'Critical' if probability[1] > 0.7 else 'High' if probability[1] > 0.5 else 'Medium' if probability[1] > 0.3 else 'Low'
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/features', methods=['GET'])
def get_features():
    try:
        features_info = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Churn' in numeric_cols:
            numeric_cols.remove('Churn')
        
        if model_features:
            for feature in model_features:
                if feature in numeric_cols:
                    features_info[feature] = {
                        'type': 'numeric',
                        'min': float(df[feature].min()),
                        'max': float(df[feature].max()),
                        'mean': float(df[feature].mean())
                    }
                elif feature in categorical_cols:
                    features_info[feature] = {
                        'type': 'categorical',
                        'options': label_encoders[feature].classes_.tolist()
                    }
        
        return jsonify(features_info), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model-info', methods=['GET'])
def model_info():
    try:
        numeric_features = [f for f in model_features if f in df.columns and f in df.select_dtypes(include=[np.number]).columns.tolist()] if model_features else []
        cat_features = [f for f in model_features if f in categorical_cols] if model_features else []
        
        return jsonify({
            'model_features': model_features,
            'total_features': len(model_features) if model_features else 0,
            'numeric_features': numeric_features,
            'categorical_features': cat_features,
            'categorical_options': {col: label_encoders[col].classes_.tolist() for col in cat_features},
            'llm_available': COHERE_API_KEY != "YOUR_COHERE_API_KEY"
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)