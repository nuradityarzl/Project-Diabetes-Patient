import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model_filename = 'Random_Forest_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

sc_X_filename = 'fitted_scaler.pkl'
with open(sc_X_filename, 'rb') as file:
    sc_X = pickle.load(file)

# Streamlit app
def main():
    st.title('Diabetes Outcome Prediction')
    st.markdown("Enter the patient's details to predict diabetes outcome.")

    # Create input fields for user
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider('Pregnancies', 0, 20, 1)
        glucose = st.slider('Glucose', 0, 200, 100)
        blood_pressure = st.slider('Blood Pressure', 0, 150, 70)
        skin_thickness = st.slider('Skin Thickness', 0, 100, 20)

    with col2:
        insulin = st.slider('Insulin', 0, 900, 100)
        bmi = st.slider('BMI', 0.0, 60.0, 25.0)
        diabetes_pedigree = st.slider('Diabetes Pedigree Function', 0.0, 2.0, 0.5)
        age = st.slider('Age', 0, 100, 30)

    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })

    # Preprocess the input data
    input_data_scaled = sc_X.transform(input_data)
    prediction = model.predict(input_data_scaled)

    if st.button('Predict'):
        # Use the model to make predictions
        prediction_text = 'Positive' if prediction[0] == 1 else 'Negative'
        prediction_color = 'red' if prediction[0] == 1 else 'green'
        st.markdown('## Prediction Result')
        prediction_result = st.empty()
        with prediction_result:
            if prediction[0] == 1:
                st.markdown('<div style="background-color:#ffcccc; padding:10px; border-radius:10px;">'
                            f'<h4 style="color:red;">{prediction_text}</h4>'
                            '</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div style="background-color:#dcffdc; padding:10px; border-radius:10px;">'
                            f'<h4 style="color:green;">{prediction_text}</h4>'
                            '</div>',
                            unsafe_allow_html=True)
                
if __name__ == '__main__':
    main()
