import streamlit as st

API_BASE = "http://localhost:8000/api/v1"


def get_auth_headers() -> dict:
    token = st.session_state.get("access_token", "")
    return {"Authorization": f"Bearer {token}"}
