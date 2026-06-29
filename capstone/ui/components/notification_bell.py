"""Streamlit notification bell — polls the API every 30 s for unread alerts."""
import time
import streamlit as st
import httpx
from frontend.utils.api import API_BASE, get_auth_headers


def render_notification_bell():
    """
    Renders a 🔔 button in the sidebar showing unread count.
    Clicking it opens an expander listing all unread notifications
    with a "Mark all read" action.

    Call this once at the top of your sidebar block.
    """
    headers = get_auth_headers()

    # Cache fetch with a 30-second TTL so we don't hammer the API
    @st.cache_data(ttl=30, show_spinner=False)
    def _fetch_unread(_headers_key: str):
        try:
            resp = httpx.get(f"{API_BASE}/notifications/unread", headers=headers, timeout=5)
            return resp.json() if resp.status_code == 200 else []
        except Exception:
            return []

    notifications = _fetch_unread(str(headers))
    count = len(notifications)

    bell_label = f"🔔 Notifications {'🔴' if count else ''} ({count})"

    with st.sidebar.expander(bell_label, expanded=False):
        if not notifications:
            st.caption("You're all caught up ✅")
            return

        if st.button("✅ Mark all as read", use_container_width=True):
            httpx.patch(f"{API_BASE}/notifications/read-all", headers=headers, timeout=5)
            st.cache_data.clear()
            st.rerun()

        for notif in notifications:
            with st.container():
                st.markdown(f"**{notif['title']}**")
                st.caption(notif["message"])
                st.caption(f"🕐 {notif.get('sent_at', '')[:16]}")
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("✓", key=f"read_{notif['id']}"):
                        httpx.patch(
                            f"{API_BASE}/notifications/{notif['id']}/read",
                            headers=headers, timeout=5
                        )
                        st.cache_data.clear()
                        st.rerun()
                st.divider()
