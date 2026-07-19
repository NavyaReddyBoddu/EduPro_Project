import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="EduPro Predictive Dashboard",
    page_icon="📚",
    layout="wide"
)

# -------------------------------
# Load Machine Learning Model
# -------------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "enrollment_model.pkl"

model = joblib.load(MODEL_PATH)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📚 EduPro")

st.sidebar.write("Predictive Analytics Dashboard")

st.sidebar.info("""
This dashboard predicts:

• Course Enrollment

• Estimated Revenue

using a trained Machine Learning model.
""")

# -------------------------------
# Main Title
# -------------------------------
st.title("📚 EduPro Predictive Analytics Dashboard")
st.markdown("### Course Demand & Revenue Forecasting")

st.markdown("---")

# -------------------------------
# User Inputs
# -------------------------------

col1, col2 = st.columns(2)

with col1:

    course_category = st.selectbox(
        "Course Category",
        [0, 1, 2, 3]
    )

    course_type = st.selectbox(
        "Course Type",
        [0, 1]
    )

    course_level = st.selectbox(
        "Course Level",
        [0, 1, 2]
    )

    course_price = st.number_input(
        "Course Price",
        min_value=0.0,
        value=1000.0
    )

with col2:

    course_duration = st.number_input(
        "Course Duration (Hours)",
        min_value=1,
        value=10
    )

    course_rating = st.slider(
        "Course Rating",
        1.0,
        5.0,
        4.0
    )

    price_band = st.selectbox(
        "Price Band",
        [0, 1, 2]
    )

st.markdown("---")

# -------------------------------
# Prediction
# -------------------------------

if st.button("🔍 Predict Enrollment"):

    input_data = pd.DataFrame({
        "CourseCategory": [course_category],
        "CourseType": [course_type],
        "CourseLevel": [course_level],
        "CoursePrice": [course_price],
        "CourseDuration": [course_duration],
        "CourseRating": [course_rating],
        "PriceBand": [price_band]
    })

    # Predict Enrollment
    prediction = model.predict(input_data)

    # Estimate Revenue
    predicted_revenue = prediction[0] * course_price

    # Display Metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="📈 Predicted Enrollment",
            value=f"{prediction[0]:.0f} Students"
        )

    with col2:
        st.metric(
            label="💰 Estimated Revenue",
            value=f"₹ {predicted_revenue:,.2f}"
        )

    st.markdown("---")

    st.subheader("📋 Input Details")

    st.dataframe(input_data)

st.markdown("---")

st.caption("EduPro Internship Project | Predictive Modeling for Course Demand and Revenue Forecasting")
