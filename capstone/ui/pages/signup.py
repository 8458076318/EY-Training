"""Signup page with full health profile collection."""
import streamlit as st
import httpx
from ui.utils.api import API_BASE


def render():
    st.header("Create your account")
    col1, col2 = st.columns(2)

    with col1:
        full_name = st.text_input("Full Name *")
        email = st.text_input("Email *")
        password = st.text_input("Password *", type="password")
        phone = st.text_input("Phone (for SMS reminders)")

    with col2:
        st.subheader("Health Profile")
        height = st.number_input("Height (cm) *", 100, 250, 170)
        weight = st.number_input("Weight (kg) *", 30, 300, 70)
        age = st.number_input("Age *", 1, 120, 25)
        gender = st.selectbox("Gender *", ["male", "female", "other"])
        profession = st.text_input("Profession *", placeholder="e.g. Software Engineer")

        diseases = st.multiselect(
            "Pre-existing Conditions",
            ["diabetes", "hypertension", "asthma", "arthritis", "thyroid", "pcod", "obesity"],
        )
        disabilities = st.multiselect(
            "Chronic Issues",
            ["BP (Blood Pressure)", "Sugar (Diabetes)", "Heart Condition"],
        )

    if st.button("Create Account", type="primary", use_container_width=True):
        payload = {
            "email": email, "password": password, "full_name": full_name,
            "phone_number": phone or None,
            "health_profile": {
                "height_cm": height, "weight_kg": weight, "age": age,
                "gender": gender, "profession": profession,
                "diseases": diseases,
                "disabilities": [d.split(" ")[0] for d in disabilities],
            },
        }
        with st.spinner("Creating account…"):
            resp = httpx.post(f"{API_BASE}/auth/signup", json=payload)
        if resp.status_code == 201:
            st.success("Account created! Please log in.")
        else:
            st.error(resp.json().get("detail", "Signup failed"))
