"""Streamlit entrypoint — multi-page app with notification bell."""
import streamlit as st

st.set_page_config(
    page_title="AI Day Planner",
    page_icon="🗓️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.utils.auth import is_authenticated
from frontend.pages import signup, login, dashboard, generate_plan, my_plans, feedback
from frontend.components.notification_bell import render_notification_bell

PAGES = {
    "🏠 Dashboard": dashboard,
    "📅 Generate Plan": generate_plan,
    "📋 My Plans": my_plans,
    "⭐ Give Feedback": feedback,
}

if not is_authenticated():
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        login.render()
    with tab2:
        signup.render()
else:
    with st.sidebar:
        st.title("🗓️ AI Day Planner")
        st.markdown("---")

        # Notification bell — polls every 30 s for unread push alerts
        render_notification_bell()

        st.markdown("---")
        selection = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

    PAGES[selection].render()
