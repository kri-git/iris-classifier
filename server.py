import pickle
import streamlit as st
import numpy as np

# Load the saved model
with open('decision_tree_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Function to predict the species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predictions = loaded_model.predict(input_data)
    labels = {
    0 : "setosa",
    1 : "versicolor",
    2 : "virginica"
    }

    predicted_labels = [labels[pred] for pred in predictions]
    return predicted_labels[0]

def main():
    st.title("Iris Flower Species Prediction")
    st.write("This is a web interface where the properties of IRIS flowers like sepal and petal properties are taken as input. Based on the input features, the species of the IRIS flower is predicted. The model is trained based on the IRIS dataset")
    st.write("Please input the measurements in centimeters.")

    # Input fields for user to input the measurements
    sepal_length = st.number_input("Sepal Length between 4.3cm and 8.0cm", min_value=4.3, max_value=8.0, value=5.0)
    sepal_width = st.number_input("Sepal Width between 2cm and 4.5cm", min_value=2.0, max_value=4.5, value=3.0)
    petal_length = st.number_input("Petal Length between 1cm and 7cm", min_value=1.0, max_value=7.0, value=5.0)
    petal_width = st.number_input("Petal Width between 0.1cm and 2.5cm", min_value=0.1, max_value=2.5, value=1.0)

    # Predict the species when the user clicks the button
    if st.button("Predict"):
        species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
        st.write(f"Predicted Species: {species}")

if __name__ == "__main__":
    main()
