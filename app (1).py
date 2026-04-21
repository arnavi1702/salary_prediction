
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 1. Load the Model ---
# The model was saved as 'best_model.pkl'
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# --- 2. Recreate and Fit Preprocessing Objects ---
# These were not saved, so we need to refit them using the original data characteristics
# For demonstration, we'll assume the original 'df' and 'X' are accessible or can be recreated.
# In a real deployment, these would ideally be saved alongside the model.

# Placeholder for original DataFrame (assuming it's available or can be loaded)
# For this example, we'll use a dummy df for categorical columns and X for scaler, based on known structure.

# Load the original data to fit the encoders and scaler
# For simplicity, we are assuming 'Salary Data.csv' is available in the Colab environment.
# In a production environment, you might load a small sample of the original data or save the fitted scalers/encoders.
original_df = pd.read_csv('/content/Salary Data.csv')

# Handle missing values as done in the notebook
for col in ['Age', 'Years of Experience', 'Salary']:
    original_df[col] = original_df[col].fillna(original_df[col].mean())
for col in ['Gender', 'Education Level', 'Job Title']:
    original_df[col] = original_df[col].fillna(original_df[col].mode()[0])

# Label Encoders
le_gender = LabelEncoder()
le_education = LabelEncoder()
le_job_title = LabelEncoder()

original_df['Gender'] = le_gender.fit_transform(original_df['Gender'])
original_df['Education Level'] = le_education.fit_transform(original_df['Education Level'])
original_df['Job Title'] = le_job_title.fit_transform(original_df['Job Title'])

# StandardScaler
X_original = original_df.drop('Salary', axis=1)
scaler = StandardScaler()
scaler.fit(X_original) # Fit scaler on the original data

# --- Streamlit App Layout --- 
st.set_page_config(page_title='Salary Predictor', layout='centered')
st.title('Predict Your Salary 💰')
st.write('Enter your details below to get an estimated salary.')

# --- User Input Features ---

st.sidebar.header('Input Your Details')

def user_input_features():
    age = st.sidebar.slider('Age', 18, 65, 30)
    gender = st.sidebar.selectbox('Gender', le_gender.classes_)
    education_level = st.sidebar.selectbox('Education Level', le_education.classes_)
    job_title = st.sidebar.selectbox('Job Title', le_job_title.classes_)
    years_of_experience = st.sidebar.slider('Years of Experience', 0, 40, 5)

    data = {
        'Age': age,
        'Gender': gender,
        'Education Level': education_level,
        'Job Title': job_title,
        'Years of Experience': years_of_experience
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display User Input
st.subheader('Your Input Parameters:')
st.write(input_df)

# --- Preprocess User Input ---
def preprocess_input(df_input):
    df_processed = df_input.copy()

    # Apply Label Encoding
    df_processed['Gender'] = le_gender.transform(df_processed['Gender'])
    df_processed['Education Level'] = le_education.transform(df_processed['Education Level'])
    df_processed['Job Title'] = le_job_title.transform(df_processed['Job Title'])

    # Scale numerical features
    # Ensure the order of columns matches the training data used for the scaler
    # X_original.columns was: ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
    df_processed_scaled = scaler.transform(df_processed[X_original.columns])
    return df_processed_scaled

processed_input = preprocess_input(input_df)

# --- Prediction ---
if st.button('Predict Salary'):
    prediction = model.predict(processed_input)
    st.success(f'The estimated salary is: ${prediction[0]:,.2f}')
