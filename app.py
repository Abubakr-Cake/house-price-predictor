import pandas as pd
import streamlit as st

st.title("California House Price Explorer 🏠")

try:
    df = pd.read_csv("housing.csv")
    st.success("✅ Dataset loaded successfully!")

    st.subheader("Raw Data Preview")
    st.write(df.head())

    st.subheader("Data Summary")
    st.write(df.describe())

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    # Drop rows with missing values
    df_cleaned = df.dropna()

    # Select features and target
    X = df_cleaned[["median_income", "total_rooms", "housing_median_age"]]
    y = df_cleaned["median_house_value"]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Show training status
    st.subheader("Model Training")
    st.success("✅ Linear Regression model trained successfully.")

    st.subheader("📈 Predict House Price")

    # Create sliders to get user input
    income = st.slider("Median Income (10,000s)", 0.0, 15.0, 5.0)
    rooms = st.slider("Total Rooms", 1, 10000, 2000)
    age = st.slider("Housing Median Age", 1, 100, 30)

    # Combine user input into a single row
    input_data = pd.DataFrame([[income, rooms, age]],
                            columns=["median_income", "total_rooms", "housing_median_age"])

    # Show user input
    st.write("User Input:")
    st.write(input_data)

    # Make prediction
    prediction = model.predict(input_data)[0]
    st.success(f"🏡 Estimated House Price: ${prediction:,.0f}")



except FileNotFoundError:
    st.error("❌ 'housing.csv' not found. Please place it in the same folder as app.py.")
except Exception as e:
    st.error(f"🚨 An unexpected error occurred: {e}")
