"""
Streamlit UI for the Multi-Agent Day Planner.
Run: streamlit run streamlit_app/app.py
"""
import streamlit as st
import httpx
from datetime import date

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="AI Day Planner", page_icon="🧠", layout="wide")

with st.sidebar:
    st.title("🧠 AI Day Planner")
    st.caption("Multi-Agent • Indian Meals • SMS Reminders")
    st.divider()
    page = st.radio("Navigate", [
        "📅 Day Planner", "🥗 Meal Plan", "⚖️ BMI & Weight", "📜 History", "📊 Analytics"
    ], label_visibility="collapsed")
    st.divider()
    user_id = st.text_input("User ID", value="user_001")
    phone   = st.text_input("Phone (SMS)", placeholder="+91XXXXXXXXXX")


def api_post(path, payload):
    try:
        r = httpx.post(f"{API_BASE}{path}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return {}


def api_get(path, params=None):
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params or {}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error: {e}")
    return {}


# ── Day Planner ───────────────────────────────────────────────────────────────
if page == "📅 Day Planner":
    st.title("📅 Your AI-Powered Day Plan")
    st.caption("GPT-4o crafts your personalised Indian wellness schedule")
    c1, c2 = st.columns(2)
    wake_time  = c1.time_input("Wake-up time")
    sleep_time = c2.time_input("Sleep time")
    wake_str   = wake_time.strftime("%H:%M") if wake_time else "06:00"
    sleep_str  = sleep_time.strftime("%H:%M") if sleep_time else "22:30"
    prefs      = st.text_area("Preferences", placeholder="e.g. vegetarian, no gym, 30 min yoga")

    if st.button("✨ Generate My Day Plan", use_container_width=True, type="primary"):
        with st.spinner("GPT-4o is planning your day…"):
            result = api_post("/planner/generate", {
                "wake_time": wake_str, "sleep_time": sleep_str,
                "phone": phone or None, "user_id": user_id,
                "preferences": {"notes": prefs} if prefs else None,
            })
        if result and result.get("events"):
            st.success(f"Plan ready for {result.get('date')} — {result.get('reminders_scheduled', 0)} SMS reminders scheduled")
            icons = {"sleep":"😴","wake":"🌅","workout":"💪","meditation":"🧘","meal":"🍽️","reading":"📖","work":"💼","break":"☕"}
            for ev in result["events"]:
                icon = icons.get(ev.get("category",""), "📌")
                with st.expander(f"{icon} {ev['time']} — {ev['activity']} ({ev.get('duration_minutes','-')} min)"):
                    st.caption(ev.get("reminder_message",""))


# ── Meal Plan ─────────────────────────────────────────────────────────────────
elif page == "🥗 Meal Plan":
    st.title("🥗 Indian Meal Plan")
    plan_date = st.date_input("Date", value=date.today())
    if plan_date.weekday() >= 5:
        st.info("🎉 Weekend — cheat meal included!")

    if st.button("🍛 Generate Meal Plan", use_container_width=True, type="primary"):
        with st.spinner("Crafting your Indian menu…"):
            result = api_post("/planner/meals", {"date": str(plan_date), "user_id": user_id})
        if result and result.get("meals"):
            meals = result["meals"]
            cols  = st.columns(3)
            for col, (mt, em) in zip(cols, [("breakfast","🌄"),("lunch","☀️"),("dinner","🌙")]):
                meal = meals.get(mt, {})
                with col:
                    st.subheader(f"{em} {mt.title()}")
                    st.markdown(f"**{meal.get('name','—')}**")
                    st.caption(f"~{meal.get('calories','-')} kcal · {meal.get('prep_time_mins','-')} min")
                    if meal.get("ingredients"):
                        st.write(", ".join(meal["ingredients"]))


# ── BMI & Weight Plan ─────────────────────────────────────────────────────────
elif page == "⚖️ BMI & Weight":
    st.title("⚖️ BMI Calculator & AI Weight Plan")

    with st.form("bmi_form"):
        c1, c2 = st.columns(2)
        name       = c1.text_input("Name", value="Arjun")
        age        = c2.number_input("Age", 10, 100, 28)
        gender     = c1.selectbox("Gender", ["male","female","other"])
        weight_kg  = c2.number_input("Weight (kg)", 30.0, 200.0, 70.0, step=0.5)
        height_cm  = c1.number_input("Height (cm)", 100.0, 220.0, 175.0, step=0.5)
        activity   = c2.selectbox("Activity level", ["sedentary","lightly_active","moderately_active","very_active"], index=2)
        submitted  = st.form_submit_button("Calculate BMI + Get Plan", type="primary", use_container_width=True)

    if submitted:
        bmi_r = api_get("/health/bmi", {"weight_kg": weight_kg, "height_cm": height_cm})
        if bmi_r:
            bmi_val = bmi_r["bmi"]
            color   = {"Normal weight":"🟢","Underweight":"🔵","Overweight":"🟡","Obese":"🔴"}.get(bmi_r["category"],"⚪")
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("BMI",        f"{bmi_val}")
            m2.metric("Category",   f"{color} {bmi_r['category']}")
            m3.metric("Goal",       bmi_r["goal"].title())
            m4.metric("Ideal wt",   f"{bmi_r['ideal_weight_kg']} kg")
            st.progress(min(max((bmi_val - 10) / 30, 0), 1), text=f"BMI scale: {bmi_val}")
            st.divider()

            with st.spinner("GPT-4o building your Indian weight plan…"):
                plan_r = api_post("/health/weight-plan", {
                    "name":name,"age":age,"gender":gender,"weight_kg":weight_kg,
                    "height_cm":height_cm,"activity_level":activity,"user_id":user_id,
                })

            if plan_r and plan_r.get("plan"):
                plan = plan_r["plan"]
                st.subheader("📋 Your Personalised Plan")
                st.info(plan.get("summary",""))
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Daily Cal",  plan.get("weekly_calorie_target","—"))
                c2.metric("Protein",    f"{plan.get('protein_g','—')}g")
                c3.metric("Carbs",      f"{plan.get('carbs_g','—')}g")
                c4.metric("Fats",       f"{plan.get('fats_g','—')}g")
                st.divider()
                ca, cb = st.columns(2)
                with ca:
                    st.subheader("🏋️ Weekly Workout")
                    for w in plan.get("weekly_workout_plan", []):
                        st.markdown(f"**{w.get('day')}** — {w.get('workout')} ({w.get('duration_mins')} min)")
                with cb:
                    st.subheader("🥗 Indian Foods")
                    st.markdown("**✅ Favour:** " + ", ".join(plan.get("indian_foods_to_favour",[])))
                    st.markdown("**❌ Avoid:** " + ", ".join(plan.get("indian_foods_to_avoid",[])))
                st.divider()
                st.subheader("🗓️ 4-Week Milestones")
                for m in plan.get("milestone_weeks", []):
                    st.markdown(f"**Week {m.get('week')}** — {m.get('expected_change_kg','?')} kg | {m.get('focus','')}")
                st.success(f"💪 {plan.get('motivational_note','Keep going!')}")


# ── History ───────────────────────────────────────────────────────────────────
elif page == "📜 History":
    st.title("📜 Agent Plan History")
    c1, c2 = st.columns([2,1])
    plan_filter = c1.selectbox("Filter", ["All","schedule","meals","bmi"])
    limit       = c2.slider("Show last N", 5, 50, 10)

    if st.button("Load History", use_container_width=True):
        params = {"limit": limit}
        if plan_filter != "All":
            params["plan_type"] = plan_filter
        result  = api_get(f"/history/{user_id}", params)
        entries = result.get("entries", [])
        if not entries:
            st.info("No history found.")
        for e in entries:
            with st.expander(f"🗂 {e['plan_type'].upper()} — {e['created_at'][:10]}"):
                st.json(e.get("payload", {}))

    if st.button("🗑️ Clear All History", type="secondary"):
        api_get(f"/history/{user_id}")
        st.warning("History cleared.")


# ── Analytics ─────────────────────────────────────────────────────────────────
elif page == "📊 Analytics":
    st.title("📊 Live Observability")
    st.markdown("""
| Dashboard | URL |
|---|---|
| Grafana (metrics + logs) | [localhost:3000](http://localhost:3000) |
| Prometheus | [localhost:9090](http://localhost:9090) |
| Loki (log queries) | [localhost:3100](http://localhost:3100) |
| FastAPI docs | [localhost:8000/docs](http://localhost:8000/docs) |
""")
    st.info("Grafana shows real-time agent calls, latency p95, SMS counts, and Loki structured logs.")
    grafana_url = st.text_input("Embed Grafana panel URL", placeholder="http://localhost:3000/d/...")
    if grafana_url:
        st.components.v1.iframe(grafana_url, height=450, scrolling=True)
