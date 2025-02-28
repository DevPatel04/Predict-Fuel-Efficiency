import streamlit as st
import numpy as np
import pandas as pd
import os
from tensorflow import keras
import tensorflow as tf

MODEL_FILE = "auto_mpg_model.keras"
TRAIN_X_FILE = "train_x.csv"
TRAIN_Y_FILE = "train_y.csv"

def load_model():
    if os.path.exists(MODEL_FILE):
        model = keras.models.load_model(MODEL_FILE)
    else:
        st.error("Model file not found.")
    return model

def save_model(model):
    model.save(MODEL_FILE)

def load_training_data():
    if os.path.exists(TRAIN_X_FILE) and os.path.exists(TRAIN_Y_FILE):
        df_X = pd.read_csv(TRAIN_X_FILE)
        df_y = pd.read_csv(TRAIN_Y_FILE)
    else:
        df_X = pd.DataFrame(columns=["displacement", "cylinders", "horsepower", "weight", "acceleration", "model_year", "origin"])
        df_y = pd.DataFrame(columns=["mpg"])
    return df_X, df_y

def save_training_data(df_X, df_y):
    df_X.to_csv(TRAIN_X_FILE, index=False)
    df_y.to_csv(TRAIN_Y_FILE, index=False)

st.title("Interactive Model Update App")

with st.form("input_form"):
    st.subheader("Enter feature values")
    displacement = st.number_input("displacement", value=150.0)
    cylinders = st.number_input("cylinders", min_value=0, value=4, step=1)
    horsepower = st.number_input("horsepower", value=100.0)
    weight = st.number_input("weight", value=2000.0)
    acceleration = st.number_input("acceleration", value=15.0)
    model_year = st.number_input("model_year",value=1970, step=1)
    origin = st.number_input("origin", min_value=0, value=1, step=1)
    submitted = st.form_submit_button("Submit Data")

if submitted:
    # Convert user input to numpy array
    x_input = np.array([[displacement, cylinders, horsepower, weight, acceleration, model_year, origin]])

    # Load model
    model = load_model()

    # Attempt prediction
    try:
        y_pred = model.predict(x_input)
    except Exception as e:
        st.error("Model prediction error: " + str(e))
        y_pred = [None]

    st.write("Predicted value:", y_pred[0])

    # Load existing training data
    df_X, df_y = load_training_data()

    # Create new rows for training data
    new_X = pd.DataFrame([[displacement, cylinders, horsepower, weight, acceleration, model_year, origin]],
                         columns=["displacement", "cylinders", "horsepower", "weight", "acceleration", "model_year", "origin"])
    new_y = pd.DataFrame([[y_pred[0]]], columns=["mpg"])
    # print(new_X)
    # print(new_y)

    # Concatenate the new data with existing training data
    df_X = pd.concat([df_X, new_X], ignore_index=True)
    df_y = pd.concat([df_y, new_y], ignore_index=True)  
    # print(df_X)
    # print(df_y)


    try:
        if "Unnamed: 0" in df_X.columns:
            df_X = df_X.drop(columns=["Unnamed: 0"])
        df_X = df_X.to_numpy().astype(np.float32)
        df_y = df_y['mpg'].to_numpy().astype(np.float32)
        # Retrain the model with the updated training data
        AUTO = tf.data.experimental.AUTOTUNE

        train_ds = (
        tf.data.Dataset
        .from_tensor_slices((df_X, df_y))
        .batch(32)
        .prefetch(AUTO)
    )
        model.fit(train_ds, epochs=10)
        save_model(model)
        save_training_data(pd.DataFrame(df_X, columns=["displacement", "cylinders", "horsepower", "weight", "acceleration", "model_year", "origin"]), pd.DataFrame(df_y, columns=["mpg"]))
        st.success("Model updated with new data.")
    except Exception as e:
        st.error("Error updating model: " + str(e))