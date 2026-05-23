import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Fuel Efficiency Predictor")
st.title("🏎️ Car MPG Predictor")

# 1. Data Generation (You can replace this with pd.read_csv)
data = {
    'Horsepower': [50, 80, 100, 130, 150, 200, 250],
    'MPG': [45, 38, 30, 25, 20, 15, 12]
}
df = pd.DataFrame(data)

# 2. Train the Model
X = df[['Horsepower']]
y = df['MPG']
model = LinearRegression()
model.fit(X, y)

# 3. Sidebar for User Input
st.sidebar.header("Car Specifications")
hp_input = st.sidebar.slider("Select Horsepower:", 50, 300, 100)

# 4. Prediction
prediction = model.predict([[hp_input]])[0]

# 5. Display Results
st.metric(label="Predicted Fuel Efficiency (MPG)", value=f"{prediction:.2f} miles/gal")

# 6. Visualization
fig, ax = plt.subplots()
ax.scatter(df['Horsepower'], df['MPG'], color='blue', label='Historical Data')
ax.plot(df['Horsepower'], model.predict(X), color='red', label='Regression Line')
ax.scatter(hp_input, prediction, color='green', s=150, label='Your Selection', zorder=5)

ax.set_xlabel("Horsepower")
ax.set_ylabel("Miles Per Gallon (MPG)")
ax.legend()
st.pyplot(fig)
