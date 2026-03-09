from flask import Flask, render_template, request, flash
import joblib
import numpy as np
import pandas as pd
import os
import random
import warnings
from sklearn.exceptions import InconsistentVersionWarning

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

def load_model(model_path: str):
    """Load model and surface version mismatch warning in a cleaner way."""
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", InconsistentVersionWarning)
            loaded_model = joblib.load(model_path)

            for warning_obj in caught_warnings:
                if issubclass(warning_obj.category, InconsistentVersionWarning):
                    print(
                        "Warning: model was trained with a different scikit-learn version. "
                        "Current predictions may be unreliable. "
                        "Recommended fix: retrain and re-save the model using the current environment."
                    )
                    break
            return loaded_model
    except FileNotFoundError:
        print("Warning: Model file not found. Using dummy predictions.")
        return None


# Load trained model with error handling
model = load_model("logreg_model.pkl")

# Mapping back numeric prediction to original stage
stage_map = {
    0: 'NORMAL',
    1: 'HYPERTENSION (Stage-1)',
    2: 'HYPERTENSION (Stage-2)',
    3: 'HYPERTENSIVE CRISIS'
}

# Medical-grade color mapping for results
color_map = {
    0: '#108981',
    1: '#F59E0B',
    2: '#F97316',
    3: '#EF4444'
}

# Detailed medical recommendations
recommendations = {
    0: {
        'title': 'Normal Blood Pressure',
        'description': 'Your cardiovascular risk assessment indicates normal blood pressure levels.',
        'actions': [
            'Maintain current healthy lifestyle',
            'Regular physical activity (150 minutes/week)',
            'Continue balanced, low-sodium diet',
            'Annual blood pressure monitoring',
            'Regular health check-ups'
        ],
        'priority': 'Low Risk'
    },
    1: {
        'title': 'Stage 1 Hypertension',
        'description': 'Mild elevation detected requiring lifestyle modifications and medical consultation.',
        'actions': [
            'Schedule appointment with healthcare provider',
            'Implement DASH diet plan',
            'Increase physical activity gradually',
            'Monitor blood pressure bi-weekly',
            'Reduce sodium intake (<2300mg/day)',
            'Consider stress management techniques'
        ],
        'priority': 'Moderate Risk'
    },
    2: {
        'title': 'Stage 2 Hypertension',
        'description': 'Significant hypertension requiring immediate medical intervention and treatment.',
        'actions': [
            'URGENT: Consult physician within 1-2 days',
            'Likely medication therapy required',
            'Comprehensive cardiovascular assessment',
            'Daily blood pressure monitoring',
            'Strict dietary sodium restriction',
            'Lifestyle modification counseling'
        ],
        'priority': 'High Risk'
    },
    3: {
        'title': 'Hypertensive Crisis',
        'description': 'CRITICAL: Dangerously elevated blood pressure requiring emergency medical care.',
        'actions': [
            'EMERGENCY: Seek immediate medical attention',
            'Call 911 if experiencing symptoms',
            'Do not delay treatment',
            'Monitor for stroke/heart attack signs',
            'Prepare current medication list',
            'Avoid physical exertion'
        ],
        'priority': 'EMERGENCY'
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Updated field names to match the React/Tailwind frontend
            required_fields = [
                'gender', 'ageGroup', 'familyHistory', 'medicalCare', 'bpMedication',
                'symptomSeverity', 'shortnessOfBreath', 'visionChanges', 'nosebleeds',
                'timeSinceDiagnosis', 'systolic', 'diastolic', 'diet'
            ]
            
            form_data = {}
            for field in required_fields:
                value = request.form.get(field)
                if not value or value == "":
                    flash(f"Please complete all required fields: {field}", "error")
                    return render_template('index.html')
                form_data[field] = value

            try:
                # Build all model features using the exact training column names.
                age_encoded = {
                    '18-34 years': 1,
                    '35-44 years': 2,
                    '45-54 years': 2,
                    '55-64 years': 3,
                    '65+ years': 4
                }.get(form_data['ageGroup'], 1)

                severity_encoded = {
                    'None': 0,
                    'Mild': 0,
                    'Moderate': 1,
                    'Severe': 2
                }.get(form_data['symptomSeverity'], 0)

                diagnosis_encoded = {
                    'N/A': 0,
                    'Less than 1 Year': 1,
                    '1-5 Years': 2,
                    '5+ Years': 3
                }.get(form_data['timeSinceDiagnosis'], 0)

                systolic_encoded = {
                    'Less than 120 mmHg (Normal)': 0,
                    '120-129 mmHg (Elevated)': 1,
                    '130-139 mmHg (Stage 1)': 2,
                    '140-180 mmHg (Stage 2)': 3,
                    'Higher than 180 mmHg (Crisis)': 3
                }.get(form_data['systolic'], 0)

                diastolic_encoded = {
                    'Less than 80 mmHg (Normal)': 0,
                    '80-89 mmHg (Stage 1)': 1,
                    '90-120 mmHg (Stage 2)': 2,
                    'Higher than 120 mmHg (Crisis)': 3
                }.get(form_data['diastolic'], 0)

                # Representative numeric pressures for the selected ranges.
                systolic_num = {
                    'Less than 120 mmHg (Normal)': 115,
                    '120-129 mmHg (Elevated)': 125,
                    '130-139 mmHg (Stage 1)': 135,
                    '140-180 mmHg (Stage 2)': 160,
                    'Higher than 180 mmHg (Crisis)': 190
                }.get(form_data['systolic'], 115)

                diastolic_num = {
                    'Less than 80 mmHg (Normal)': 75,
                    '80-89 mmHg (Stage 1)': 85,
                    '90-120 mmHg (Stage 2)': 105,
                    'Higher than 120 mmHg (Crisis)': 130
                }.get(form_data['diastolic'], 75)

                encoded_features = {
                    'Gender': 0 if form_data['gender'] == 'Male' else 1,
                    'Age': age_encoded,
                    'History': 1 if form_data['familyHistory'] == 'Yes' else 0,
                    'Patient': 1 if form_data['medicalCare'] == 'Yes' else 0,
                    'TakeMedication': 1 if form_data['bpMedication'] == 'Yes' else 0,
                    'Severity': severity_encoded,
                    'BreathShortness': 1 if form_data['shortnessOfBreath'] == 'Yes' else 0,
                    'VisualChanges': 1 if form_data['visionChanges'] == 'Yes' else 0,
                    'NoseBleeding': 1 if form_data['nosebleeds'] == 'Yes' else 0,
                    'Whendiagnoused': diagnosis_encoded,
                    'Systolic': systolic_encoded,
                    'Diastolic': diastolic_encoded,
                    'ControlledDiet': 1 if form_data['diet'] in ['Yes', 'Sometimes'] else 0,
                    'Systolic_num': systolic_num,
                    'Diastolic_num': diastolic_num
                }
            except KeyError as e:
                flash(f"Invalid selection detected: {str(e)}", "error")
                return render_template('index.html')

            expected_columns = list(getattr(model, "feature_names_in_", encoded_features.keys()))
            input_df = pd.DataFrame([[encoded_features[col] for col in expected_columns]], columns=expected_columns)

            if model is not None:
                prediction = model.predict(input_df)[0]
                try:
                    confidence = max(model.predict_proba(input_df)[0]) * 100
                except:
                    confidence = 85.0
            else:
                prediction = random.randint(0, 3)
                confidence = 87.5
                flash("Demo Mode: Using simulated AI prediction for demonstration", "info")

            print("Encoded Input:", encoded_features)
            
            return render_template('index.html',
                                   prediction_text=stage_map[prediction],
                                   result_color=color_map[prediction],
                                   confidence=round(confidence, 1),
                                   recommendation=recommendations[prediction],
                                   form_data=form_data)

    except Exception as e:
        print(f"Error: {str(e)}")
        flash("System error occurred. Please try again or contact technical support.", "error")
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
