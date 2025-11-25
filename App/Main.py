import streamlit as st
import sys
import os

# Import your inference pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.Inference_pipeline import predict_price_range


# ---------------------- STREAMLIT PAGE SETUP ----------------------
st.set_page_config(page_title="CodeX Beverage: Price Prediction", layout="wide")

# Title
st.markdown(
    """
    <h1 style="text-align:center; font-size:45px; font-weight:700;">
        CodeX Beverage: Price Prediction
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("")  # spacing


# ---------------------- INPUT FORM ----------------------
with st.form("price_form"):
    col1, col2, col3, col4 = st.columns(4)

    # ---- COL 1 ----
    with col1:
        age = st.slider("Age", 18, 70, 30)

        income_levels = st.selectbox(
            "Income Level (In L)",
            ["<10L", "10L - 15L", "16L - 25L", "26L - 35L", "> 35L"],
            index=None,
            placeholder="Select Income Level"
        )

        awareness_other = st.selectbox(
            "Awareness of other brands",
            ["0 to 1", "2 to 4", "above 4"],
            index=None,
            placeholder="Select Awareness Level"
        )

        packaging_preference = st.selectbox(
            "Packaging Preference",
            ["Eco-Friendly","Simple", "Premium"],
            index=None,
            placeholder="Select Packaging"
        )

    # ---- COL 2 ----
    with col2:
        gender = st.selectbox(
            "Gender",
            ["M", "F"],
            index=None,
            placeholder="Select Gender"
        )

        consume_freq = st.selectbox(
            "Consume Frequency (weekly)",
            ["0-2 times", "3-4 times", "5-7 times"],
            index=None,
            placeholder="Select Frequency"
        )

        reasons = st.selectbox(
            "Reasons for choosing brands",
            ["availability","price", "quality", "brand reputation"],
            index=None,
            placeholder="Select Reason"
        )

        health_concerns = st.selectbox(
            "Health Concerns",
            ["Low (Not very concerned)",
             "Medium (Moderately health-conscious)",
             "High (Very health-conscious)"
             ],
            index=None,
            placeholder="Select Health Concern"
        )

    # ---- COL 3 ----
    with col3:
        zone = st.selectbox(
            "Zone",
            ["rural", "semi-urban", "urban", "metro"],
            index=None,
            placeholder="Select Zone"
        )

        current_brand = st.selectbox(
            "Current Brand",
            ["established", "newcomer"],
            index=None,
            placeholder="Select Brand Type"
        )

        flavor_preference = st.selectbox(
            "Flavor Preference",
            ["traditional", "Exotic"],
            index=None,
            placeholder="Select Flavor"
        )

        purchase_channel = st.selectbox(
            "Purchase Channel",
            ["online", "retail store"],
            index=None,
            placeholder="Select Channel"
        )

    # ---- COL 4 ----
    with col4:
        occupation = st.selectbox(
            "Occupation",
            ["working professional", "student", "entrepreneur", "retired"],
            index=None,
            placeholder="Select Occupation"
        )

        preferable_size = st.selectbox(
            "Preferable Consumption Size",
            ["Small (250 ml)", "Medium (500 ml)", "Large (1 L)"],
            index=None,
            placeholder="Select Size"
        )

        typical_situations = st.selectbox(
            "Typical Consumption Situations",
            ["Casual (eg. At home)",
             "Social (eg. Parties)",
             "Active (eg. Sports, gym)"
             ],
            index=None,
            placeholder="Select Situation"
        )

    st.write("")
    col_left, col_center, col_right = st.columns([2, 1, 2])

    with col_center:
        submitted = st.form_submit_button(
            "Calculate Price Range",
            use_container_width=False
        )

# ---------------------- PROCESS ON SUBMIT ----------------------
if submitted:

    # Convert Age → Age Group (same logic as training)
    if age <= 25:
        age_group = "18-25"
    elif age <= 35:
        age_group = "26-35"
    elif age <= 45:
        age_group = "36-45"
    elif age <= 55:
        age_group = "46-55"
    elif age <= 70:
        age_group = "56-70"
    else:
        age_group = "70+"


    # 🔍 Logical Validation Checks

    if occupation == "student" and age > 50:
        st.error("❌ Invalid Input: A student cannot realistically be above age 50.")
        st.stop()

    if occupation == "retired" and age < 40:
        st.error("❌ Invalid Input: Retired individuals are usually 40+.")
        st.stop()

    input_dict = {
        "gender": gender,
        "zone": zone,
        "occupation": occupation,
        "income_levels": income_levels,
        "consume_frequency(weekly)": consume_freq,
        "preferable_consumption_size": preferable_size,
        "health_concerns": health_concerns,
        "age_group": age_group,
        "current_brand": current_brand,
        "awareness_of_other_brands": awareness_other,
        "reasons_for_choosing_brands": reasons,
        "flavor_preference": flavor_preference,
        "purchase_channel": purchase_channel,
        "packaging_preference": packaging_preference,
        "typical_consumption_situations": typical_situations,
        "cf_ab_score": 0,
        "zas_score": 0,
        "bsi": 0
    }

    # Validation check for empty selections
    required_fields = {
        "Gender": gender,
        "Zone": zone,
        "Occupation": occupation,
        "Income Level": income_levels,
        "Weekly Consumption Frequency": consume_freq,
        "Preferred Size": preferable_size,
        "Health Concerns": health_concerns,
        "Current Brand": current_brand,
        "Awareness of Other Brands": awareness_other,
        "Reason for Choosing Brand": reasons,
        "Flavor Preference": flavor_preference,
        "Purchase Channel": purchase_channel,
        "Packaging Preference": packaging_preference,
        "Typical Consumption Situation": typical_situations,
    }

    missing = [key for key, value in required_fields.items() if value is None]

    if missing:
        st.error(f"❌ Please fill these required fields: {', '.join(missing)}")
        st.stop()

    # Run prediction
    prediction = predict_price_range(input_dict)

    # Output
    st.success(f"### 🥤 Predicted Price Range: **{prediction}**")

