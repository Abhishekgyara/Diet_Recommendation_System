```markdown
# 🥗 AI Diet Recommendation System

A modern, elegant Streamlit web application that uses machine learning to provide personalized diet recommendations based on patient health metrics, lifestyle factors, and disease conditions.

## ✨ Features

- **ML-Powered Predictions** - Trained model recommends optimal diet plans
- **Interactive Dashboard** - Clean, responsive UI with metric cards
- **Patient Input** - Comprehensive health parameters collection
- **Smart Visualizations** - Interactive Plotly charts (bar, pie, radar)
- **AI Recommendations** - Gemini AI for personalized diet plans and lifestyle tips
- **Real-time Predictions** - Instant diet recommendations with visual feedback

## 🛠️ Tech Stack

- Streamlit (Frontend)
- Scikit-learn & XGBoost (Machine Learning)
- Plotly (Visualization)
- Google Gemini AI (AI Recommendations)

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-diet-recommendation.git
cd ai-diet-recommendation
```

### 2. Set Up Gemini API Key

Create a `.env` file in the project root and add your Gemini API key:

```
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

> **Note:** Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### 3. Place Required Files in Project Root

Ensure these files are in the same directory as `app.py`:

- `diet_model.pkl` - Trained ML model
- `scaler.pkl` - Fitted StandardScaler
- `diet_recommendations_dataset.csv` - Dataset file

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📹 Demo Execution Video

### Step-by-Step Walkthrough:

1. **App Launch** - The app opens with a beautiful gradient header and sidebar navigation
2. **Patient Input** - Fill in patient details in the sidebar:
   - Age, Gender, Weight, Height, BMI
   - Disease Type, Severity, Activity Level
   - Health metrics (Cholesterol, Blood Pressure, Glucose)
   - Dietary preferences, Allergies, Cuisine preferences
   - Exercise hours, Adherence score, Nutrient imbalance score
3. **Click "Predict Diet"** - The ML model processes the input
4. **Results Display**:
   - Large, colorful card shows recommended diet
   - Patient summary expander shows key metrics
   - Interactive Plotly visualizations (bar, pie, radar charts)
   - Gemini AI generates personalized recommendations displayed in tabs:
     - 💡 AI Suggestions
     - 🍽️ Meal Plan
     - 🏃 Lifestyle Tips

### Screenshots Preview

```
🏠 Main Dashboard
├── Gradient Header: "AI Diet Recommendation System"
├── Sidebar: All patient input fields
├── Metric Cards: BMI, Calories, Exercise Hours, Age
└── Footer: "Machine Learning Powered Diet Recommendation System"

📊 Prediction Results
├── 🥗 Recommended Diet (Large highlighted card)
├── 📋 Patient Summary (Expandable)
├── 📊 Health Metrics Visualization
│   ├── Bar Chart (BMI, Calories, Exercise)
│   ├── Pie Chart (Health metrics distribution)
│   └── Radar Chart (BMI, Cholesterol, Glucose, Calories)
└── 🤖 AI-Powered Recommendations
    ├── 💡 AI Suggestions Tab
    ├── 🍽️ Meal Plan Tab
    └── 🏃 Lifestyle Tips Tab
```

### Demo GIF Preview

```
[Streamlit App Execution Flow]

User Inputs → Click Predict → Model Processes → Show Results
     ↓              ↓              ↓                ↓
  Sidebar      Predict Button   ML Pipeline   Diet Card + Charts + AI
```

---

## 📁 Project Structure

```
ai-diet-recommendation/
│
├── app.py                          # Main Streamlit application
├── diet_model.pkl                  # Trained ML model
├── scaler.pkl                      # Fitted StandardScaler
├── diet_recommendations_dataset.csv # Dataset
├── .env                            # Gemini API key (create this)
└── README.md                       # Project documentation
```

---

## 🧠 Machine Learning Pipeline

- **Preprocessing**: Handling missing values, encoding categorical variables, StandardScaler
- **Models Compared**: Logistic Regression, KNN, Decision Tree, Random Forest, SVM, XGBoost
- **Best Model**: Saved as `diet_model.pkl` with optimal hyperparameters
- **Prediction**: User input → Scale → Predict → Decode to original labels

---

## 🔑 API Keys Needed

- **Gemini API Key**: Required for AI recommendations
  - Get it from: [Google AI Studio](https://makersuite.google.com/app/apikey)

---

```