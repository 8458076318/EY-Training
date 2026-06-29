import streamlit as st


def is_authenticated() -> bool:
    return bool(st.session_state.get("access_token"))


def require_auth():
    if not is_authenticated():
        st.warning("Please log in to continue.")
        st.stop()
