# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load models and encoders
@st.cache_resource
def load_models():
    try:
        model = load_model('data/models/ann_model.h5')
        with open('data/models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('data/models/onehot_encoder_geography.pkl', 'rb') as f:
            onehot_encoder = pickle.load(f)
        with open('data/models/label_encoder_gender.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, scaler, onehot_encoder, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# Prediction function
def predict_churn(input_data, model, scaler, onehot_encoder, label_encoder):
    try:
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        input_df['Gender'] = label_encoder.transform(input_df['Gender'])
        
        # One-hot encode Geography
        geography_encoded = onehot_encoder.transform(input_df[['Geography']]).toarray()
        geography_df = pd.DataFrame(
            geography_encoded, 
            columns=onehot_encoder.get_feature_names_out(['Geography'])
        )
        
        # Combine with original data
        input_df = pd.concat([input_df, geography_df], axis=1)
        input_df.drop('Geography', axis=1, inplace=True)
        
        # Ensure correct column order (same as training)
        expected_columns = [
            'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
            'Geography_France', 'Geography_Germany', 'Geography_Spain'
        ]
        input_df = input_df[expected_columns]
        
        # Scale the data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        probability = prediction[0][0]
        
        return probability
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Main app
def main():
    st.title("🏦 Customer Churn Prediction App")
    st.markdown("""
    This app predicts the likelihood of a customer churning using an Artificial Neural Network.
    Please fill in the customer details below and click 'Predict' to see the results.
    """)
    
    # Load models
    model, scaler, onehot_encoder, label_encoder = load_models()
    
    if model is None:
        st.error("Failed to load models. Please check the file paths.")
        return
    
    # Create two columns for input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Customer Information")
        
        credit_score = st.slider("Credit Score", 300, 850, 650)
        age = st.slider("Age", 18, 100, 40)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
        estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 60000.0)
    
    with col2:
        st.header("Account Details")
        
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        num_products = st.slider("Number of Products", 1, 4, 2)
        has_cc = st.radio("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active = st.radio("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    
    # Create input dictionary
    input_data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_cc,
        'IsActiveMember': is_active,
        'EstimatedSalary': estimated_salary
    }
    
    # Prediction button
    if st.button("🔮 Predict Churn Probability", type="primary"):
        with st.spinner("Making prediction..."):
            probability = predict_churn(input_data, model, scaler, onehot_encoder, label_encoder)
            
            if probability is not None:
                st.success("Prediction completed!")
                
                # Display results
                col3, col4 = st.columns(2)
                
                with col3:
                    st.metric("Churn Probability", f"{probability:.2%}")
                    
                    # Progress bar
                    st.progress(float(probability))
                    
                with col4:
                    if probability > 0.5:
                        st.error(f"🚨 High Risk: Customer is likely to churn")
                        st.info("Recommended action: Offer retention incentives")
                    else:
                        st.success(f"✅ Low Risk: Customer is not likely to churn")
                        st.info("Recommended action: Continue current engagement")
                
                # Show input data summary
                st.subheader("Input Summary")
                summary_df = pd.DataFrame.from_dict(input_data, orient='index', columns=['Value'])
                st.dataframe(summary_df, use_container_width=True)

# Sidebar with information
st.sidebar.title("About")
st.sidebar.info("""
This app uses an Artificial Neural Network trained on customer data to predict churn probability.

**Features used:**
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Ownership
- Active Membership Status
- Estimated Salary
""")

st.sidebar.divider()
st.sidebar.caption("Built with Streamlit & TensorFlow")

if __name__ == "__main__":
    main()