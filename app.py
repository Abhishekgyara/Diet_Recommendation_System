"""
AI Diet Recommendation System
A modern Streamlit web application for personalized diet recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from dotenv import load_dotenv
import os
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Diet Recommendation System",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        color: #6c757d;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and scaler
@st.cache_resource
def load_models():
    """Load trained model and scaler."""
    try:
        model = joblib.load('diet_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure diet_model.pkl and scaler.pkl are in the correct directory.")
        return None, None

# Load dataset for label encoding reference
@st.cache_data
def load_dataset():
    """Load dataset to get label encoder."""
    try:
        df = pd.read_csv('diet_recommendations_dataset.csv')
        # Fit label encoder on original diet recommendations
        le = LabelEncoder()
        le.fit(df['Diet_Recommendation'])
        return le
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure diet_recommendations_dataset.csv is in the correct directory.")
        return None

def create_sample_array(inputs):
    """Create prediction sample with patient_id=0."""
    # Column order from training data
    columns = [
        'Patient_ID', 'Age', 'Gender', 'Weight_kg', 'Height_cm', 'BMI',
        'Disease_Type', 'Severity', 'Physical_Activity_Level', 
        'Daily_Caloric_Intake', 'Cholesterol_mg/dL', 'Blood_Pressure_mmHg',
        'Glucose_mg/dL', 'Dietary_Restrictions', 'Allergies',
        'Preferred_Cuisine', 'Weekly_Exercise_Hours', 
        'Adherence_to_Diet_Plan', 'Dietary_Nutrient_Imbalance_Score'
    ]
    
    # Create array with patient_id=0
    sample = [0]  # Patient_ID
    sample.extend(inputs)  # Add all other features
    
    return np.array(sample).reshape(1, -1)

def get_gemini_recommendations(age, bmi, disease, activity_level, predicted_diet):
    """Get AI-powered recommendations using Gemini."""
    try:
        # Configure Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        prompt = f"""
        Based on the following patient information:
        - Age: {age}
        - BMI: {bmi}
        - Disease Type: {disease}
        - Activity Level: {activity_level}
        - Predicted Diet: {predicted_diet}
        
        Provide a comprehensive diet recommendation including:
        1. Foods to Eat (specific foods)
        2. Foods to Avoid (specific foods)
        3. One-Day Meal Plan (breakfast, lunch, dinner, snacks)
        4. Lifestyle Recommendations
        5. Hydration Advice
        6. Exercise Suggestions
        
        Format the response with clear sections and bullet points.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating AI recommendations: {str(e)}"

def main():
    # Header
    st.markdown('<div class="main-header"><h1>🥗 AI Diet Recommendation System</h1><p>Personalized Diet Plans Powered by Machine Learning</p></div>', unsafe_allow_html=True)
    
    # Load models
    model, scaler = load_models()
    label_encoder = load_dataset()
    
    if model is None or scaler is None or label_encoder is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Patient Information")
        st.markdown("---")
        
        # Input fields
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.0)
        
        disease = st.selectbox("Disease Type", [
            "None", "Diabetes", "Hypertension", "Obesity", 
            "Heart Disease", "Thyroid", "Anemia", "Arthritis"
        ])
        
        severity = st.selectbox("Severity", ["Mild", "Moderate", "Severe"])
        activity_level = st.selectbox("Physical Activity Level", ["Sedentary", "Light", "Moderate", "Active", "Very Active"])
        calories = st.number_input("Daily Caloric Intake", min_value=1000, max_value=5000, value=2000)
        
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
        glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=400, value=100)
        
        dietary_restrictions = st.selectbox("Dietary Restrictions", ["None", "Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Keto", "Low-Carb", "Low-Fat"])
        allergies = st.selectbox("Allergies", ["None", "Nuts", "Dairy", "Shellfish", "Gluten", "Eggs", "Soy"])
        cuisine = st.selectbox("Preferred Cuisine", ["American", "Italian", "Mexican", "Indian", "Chinese", "Japanese", "Mediterranean"])
        
        exercise_hours = st.number_input("Weekly Exercise Hours", min_value=0.0, max_value=30.0, value=5.0)
        adherence = st.selectbox("Adherence to Diet Plan", ["Low", "Medium", "High"])
        nutrient_score = st.number_input("Dietary Nutrient Imbalance Score", min_value=0.0, max_value=10.0, value=5.0)
        
        st.markdown("---")
        predict_button = st.button("🎯 Predict Diet", use_container_width=True)
    
    # Main content
    if predict_button:
        with st.spinner("Analyzing patient data..."):
            # Prepare input for prediction
            # Convert categorical variables to numeric codes (same as during training)
            gender_encoded = 1 if gender == "Male" else 0
            disease_encoded = ["None", "Diabetes", "Hypertension", "Obesity", "Heart Disease", "Thyroid", "Anemia", "Arthritis"].index(disease)
            severity_encoded = ["Mild", "Moderate", "Severe"].index(severity)
            activity_encoded = ["Sedentary", "Light", "Moderate", "Active", "Very Active"].index(activity_level)
            dietary_encoded = ["None", "Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Keto", "Low-Carb", "Low-Fat"].index(dietary_restrictions)
            allergy_encoded = ["None", "Nuts", "Dairy", "Shellfish", "Gluten", "Eggs", "Soy"].index(allergies)
            cuisine_encoded = ["American", "Italian", "Mexican", "Indian", "Chinese", "Japanese", "Mediterranean"].index(cuisine)
            adherence_encoded = ["Low", "Medium", "High"].index(adherence)
            
            # Create input array
            inputs = [
                age, gender_encoded, weight, height, bmi,
                disease_encoded, severity_encoded, activity_encoded,
                calories, cholesterol, blood_pressure, glucose,
                dietary_encoded, allergy_encoded, cuisine_encoded,
                exercise_hours, adherence_encoded, nutrient_score
            ]
            
            sample = create_sample_array(inputs)
            
            # Scale features
            sample_scaled = scaler.transform(sample)
            
            # Make prediction
            prediction = model.predict(sample_scaled)
            diet_recommendation = label_encoder.inverse_transform(prediction)[0]
            
            # Display prediction
            st.markdown(f"""
                <div class="prediction-card">
                    <h2>🥗 Recommended Diet</h2>
                    <h1 style="font-size: 3rem;">{diet_recommendation}</h1>
                </div>
            """, unsafe_allow_html=True)
            
            # Patient Summary
            with st.expander("📋 Patient Summary", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Age", f"{age} years")
                    st.metric("Weight", f"{weight} kg")
                    st.metric("Height", f"{height} cm")
                with col2:
                    st.metric("BMI", f"{bmi:.1f}")
                    st.metric("Blood Pressure", f"{blood_pressure} mmHg")
                    st.metric("Glucose", f"{glucose} mg/dL")
                with col3:
                    st.metric("Calories", f"{calories} kcal/day")
                    st.metric("Exercise", f"{exercise_hours} hrs/week")
                    st.metric("Disease", disease)
            
            # Visualizations
            st.subheader("📊 Health Metrics Visualization")
            
            # Create three columns for charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                metrics_data = pd.DataFrame({
                    'Metric': ['BMI', 'Calories', 'Exercise Hours'],
                    'Value': [bmi, calories/100, exercise_hours],
                    'Unit': ['kg/m²', '×100 kcal', 'hours']
                })
                
                fig_bar = px.bar(
                    metrics_data,
                    x='Metric',
                    y='Value',
                    title='Health Metrics Overview',
                    color='Metric',
                    text='Value'
                )
                fig_bar.update_traces(textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Radar chart
                radar_data = pd.DataFrame({
                    'Metric': ['BMI', 'Cholesterol', 'Glucose', 'Calories'],
                    'Value': [bmi/25, cholesterol/200, glucose/100, calories/2000],
                    'Category': ['Patient', 'Patient', 'Patient', 'Patient']
                })
                
                fig_radar = px.line_polar(
                    radar_data,
                    r='Value',
                    theta='Metric',
                    line_close=True,
                    title='Health Metrics Radar'
                )
                fig_radar.update_traces(fill='toself')
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Pie chart (show distribution of health metrics)
            metrics_for_pie = pd.DataFrame({
                'Metric': ['BMI', 'Cholesterol', 'Glucose', 'Calories'],
                'Percentage': [bmi/50*100, cholesterol/400*100, glucose/400*100, calories/5000*100]
            })
            
            fig_pie = px.pie(
                metrics_for_pie,
                values='Percentage',
                names='Metric',
                title='Health Metrics Distribution',
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # AI Recommendations
            st.subheader("🤖 AI-Powered Recommendations")
            
            with st.spinner("Generating personalized recommendations..."):
                gemini_response = get_gemini_recommendations(
                    age, bmi, disease, activity_level, diet_recommendation
                )
                
                # Parse and display response in tabs
                tabs = st.tabs(["💡 AI Suggestions", "🍽️ Meal Plan", "🏃 Lifestyle Tips"])
                
                with tabs[0]:
                    st.markdown(gemini_response)
                
                with tabs[1]:
                    st.markdown("### One-Day Meal Plan")
                    st.markdown("""
                    **Breakfast** 🍳
                    - Oatmeal with fruits and nuts
                    - Green tea
                    
                    **Lunch** 🥗
                    - Grilled chicken salad
                    - Quinoa
                    
                    **Dinner** 🍲
                    - Baked salmon
                    - Steamed vegetables
                    - Brown rice
                    
                    **Snacks** 🥜
                    - Greek yogurt
                    - Fresh fruits
                    """)
                
                with tabs[2]:
                    st.markdown("### Lifestyle Recommendations")
                    st.markdown("""
                    🏋️ **Exercise**
                    - 30 minutes of moderate exercise daily
                    - Include both cardio and strength training
                    
                    💧 **Hydration**
                    - Drink 2-3 liters of water daily
                    - Include herbal teas
                    
                    😴 **Sleep**
                    - 7-8 hours of quality sleep
                    - Maintain consistent sleep schedule
                    
                    🧘 **Stress Management**
                    - Practice meditation
                    - Deep breathing exercises
                    """)
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p>Machine Learning Powered Diet Recommendation System | Built with Streamlit</p>
            <p style="font-size: 0.8rem;">© 2026 AI Diet Recommendation System | All Rights Reserved</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()