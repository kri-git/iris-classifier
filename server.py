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
    st.write("This is a web interface where the properties of IRIS flowers like sepal and petal properties are taken as input. Based on the input features, the species of the IRIS flower is predicted.The model is trained based on the IRIS dataset")    
    st.write("Please adjust the slider and click on predict to get the species prediction.")

    # Input fields for user to input the measurements
    sepal_length = st.slider("Sepal Length in cm", 4.3, 7.9, 5.0)
    sepal_width = st.slider("Sepal Width in cm", 2.0, 4.4, 3.0)
    petal_length = st.slider("Petal Length in cm", 1.0, 6.9, 5.0)
    petal_width = st.slider("Petal Width in cm", 0.1, 2.5, 0.4)

    # Predict the species when the user clicks the button
    if st.button("Predict"):
        species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
        st.write(f"Predicted Species: {species}")

if __name__ == "__main__":
    main()

