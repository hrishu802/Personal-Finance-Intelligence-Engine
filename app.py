import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

from utils.preprocessing import load_and_clean_data, feature_engineering, aggregate_monthly_data, get_financial_health_score, get_risk_score
from utils.recommender import generate_recommendations, ai_advisor_response, simulate_savings
from models.predictor import ExpensePredictor, detect_anomalies
from utils.insights import generate_smart_insights, what_changed_analysis, get_kpi_explanation
from utils.db import init_db, load_transactions_from_db, save_budget, get_all_budgets

# --- PAGE CONFIG ---
st.set_page_config(page_title="PFIE V4 - Enterprise FinTech", layout="wide", page_icon="🏦", initial_sidebar_state="expanded")

# --- INITIALIZE DATABASE ---
init_db()
USER_ID = "USER_101" # Default user for MVP

# --- CUSTOM PREMIUM CSS ---
st.markdown("""
<style>
    /* Global Theme Variables */
    :root {
        --bg-dark: #0B0E14;
        --card-bg: rgba(22, 27, 34, 0.6);
        --card-border: rgba(255, 255, 255, 0.08);
        --text-main: #E2E8F0;
        --text-muted: #94A3B8;
        --accent-teal: #14F1D9;
        --accent-blue: #3B82F6;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
    }
    
    /* Animations */
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Background & App */
    .stApp {
        background: radial-gradient(circle at top left, #1a1f35, var(--bg-dark) 40%);
        color: var(--text-main);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(11, 14, 20, 0.95);
        border-right: 1px solid var(--card-border);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        animation: fadeSlideIn 0.6s ease-out forwards;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    /* KPI Card Specifics */
    .kpi-title {
        color: var(--text-muted);
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .kpi-value {
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 4px;
        background: linear-gradient(90deg, #fff, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .kpi-subtext {
        font-size: 0.85rem;
        display: flex;
        flex-direction: column;
        gap: 2px;
        margin-top: 10px;
    }
    
    .kpi-subtext-main {
        display: flex;
        gap: 6px;
        align-items: center;
    }
    
    .kpi-subtext-explain {
        color: #64748B;
        font-size: 0.75rem;
        font-style: italic;
    }
    
    .trend-up { color: var(--danger); font-weight: 600; }
    .trend-down { color: var(--success); font-weight: 600; }
    .trend-neutral { color: var(--text-muted); font-weight: 600; }
    
    /* Section Titles */
    .premium-title {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, var(--accent-teal), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        margin-top: 1rem;
    }
    
    /* Chat Bubbles */
    .chat-bubble-ai {
        background: linear-gradient(135deg, rgba(20, 241, 217, 0.1), rgba(59, 130, 246, 0.1));
        border: 1px solid rgba(20, 241, 217, 0.2);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING & CACHING ---
@st.cache_data
def load_pfie_data():
    df = load_transactions_from_db()
    if df.empty:
        from utils.data_gen import generate_synthetic_data
        df = generate_synthetic_data()
        
    df['date'] = pd.to_datetime(df['date'])
    df = feature_engineering(df)
    monthly_summary = aggregate_monthly_data(df)
    return df, monthly_summary

@st.cache_resource
def get_advanced_predictor_model():
    return ExpensePredictor()

df, monthly_summary = load_pfie_data()
predictor = get_advanced_predictor_model()

# The advanced predictor takes the raw df
if len(monthly_summary) >= 4:
    predictor.train(df)

# --- SESSION STATE INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR & FILTERS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2855/2855661.png", width=60)
    st.title("PFIE Engine")
    st.markdown("<p style='color:#94A3B8; font-size: 0.9em; margin-top:-15px;'>Enterprise FinTech SaaS</p>", unsafe_allow_html=True)
    
    st.markdown("### Navigation")
    page = st.radio("", ["Overview Dashboard", "Deep Analysis", "Budgets & Control", "AI Advisor & Chat"])
    
    st.markdown("---")
    st.markdown("### Global Settings")
    assumed_income = st.number_input("Monthly Income (₹)", min_value=10000, value=80000, step=5000)
    
    st.markdown("### Filters")
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    
    all_cats = df['category'].unique().tolist()
    selected_cats = st.multiselect("Categories", all_cats, default=all_cats)
    
    all_modes = df['payment_mode'].unique().tolist()
    selected_modes = st.multiselect("Payment Modes", all_modes, default=all_modes)

# Apply Filters
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
else:
    filtered_df = df.copy()

filtered_df = filtered_df[filtered_df['category'].isin(selected_cats)]
filtered_df = filtered_df[filtered_df['payment_mode'].isin(selected_modes)]

if not filtered_df.empty:
    filtered_monthly = aggregate_monthly_data(filtered_df)
else:
    filtered_monthly = monthly_summary.copy()

# --- HELPER FUNCTIONS ---
def get_mom_change(monthly_df):
    if len(monthly_df) < 2:
        return 0, 0, "No data"
    curr = monthly_df.iloc[-1]['total_spending']
    prev = monthly_df.iloc[-2]['total_spending']
    diff = curr - prev
    perc = (diff / prev) * 100 if prev > 0 else 0
    return curr, diff, perc

# --- PAGE: OVERVIEW ---
if page == "Overview Dashboard":
    st.markdown('<div class="premium-title">Financial Overview</div>', unsafe_allow_html=True)
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        st.stop()
        
    # KPI CARDS
    col1, col2, col3 = st.columns(3)
    
    curr_spend, spend_diff, spend_perc = get_mom_change(filtered_monthly)
    trend_class = "trend-up" if spend_perc > 0 else "trend-down" if spend_perc < 0 else "trend-neutral"
    trend_icon = "↑" if spend_perc > 0 else "↓" if spend_perc < 0 else "−"
    trend_text = "Increase" if spend_perc > 0 else "Decrease" if spend_perc < 0 else "Change"
    
    kpi_explain = "Not enough data"
    if len(filtered_monthly) >= 2:
        curr_m = filtered_monthly.iloc[-1]['year_month']
        prev_m = filtered_monthly.iloc[-2]['year_month']
        kpi_explain = get_kpi_explanation(filtered_df, curr_m, prev_m)
    
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-title">Total Spend (This Month)</div>
            <div class="kpi-value">₹{curr_spend:,.0f}</div>
            <div class="kpi-subtext">
                <div class="kpi-subtext-main">
                    <span class="{trend_class}">{trend_icon} {abs(spend_perc):.1f}%</span> 
                    <span style="color:#94A3B8;">{trend_text} vs Last Month</span>
                </div>
                <div class="kpi-subtext-explain">↳ {kpi_explain}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    health_score, health_breakdown = get_financial_health_score(filtered_df, assumed_income)
    health_color = "#10B981" if health_score >= 70 else "#F59E0B" if health_score >= 40 else "#EF4444"
    health_label = "Excellent" if health_score >= 80 else "Good" if health_score >= 60 else "Risky"
    
    with col2:
        st.markdown(f"""
        <div class="glass-card" style="border-top: 3px solid {health_color};">
            <div class="kpi-title">Health Score</div>
            <div class="kpi-value">{health_score}/100</div>
            <div class="kpi-subtext">
                <div class="kpi-subtext-main">
                    <span style="color:{health_color}; font-weight:600;">{health_label}</span>
                    <span style="color:#94A3B8;"> | Stability: {health_breakdown['stability']}</span>
                </div>
                <div class="kpi-subtext-explain">↳ Driven by {health_breakdown['savings_ratio']*100:.0f}% savings & {health_breakdown['luxury_ratio']*100:.0f}% luxury ratio</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    risk_label, risk_score, risk_factors = get_risk_score(filtered_df, filtered_monthly, assumed_income)
    risk_color = "#EF4444" if risk_score > 60 else "#F59E0B" if risk_score > 30 else "#10B981"
    
    with col3:
        st.markdown('<div class="glass-card" style="border-top: 3px solid {}; padding-bottom: 5px;">'.format(risk_color), unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Risk Engine</div>', unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': risk_color},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.1)"},
                    {'range': [30, 60], 'color': "rgba(245, 158, 11, 0.1)"},
                    {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.1)"}
                ]
            }
        ))
        fig_gauge.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=120, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown(f'<div class="kpi-subtext-explain" style="text-align:center; margin-top:-10px; margin-bottom:10px;">↳ {risk_label}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # SMART INSIGHTS & VISUALS
    col_v1, col_v2 = st.columns([2, 1])
    
    with col_v1:
        st.markdown('<h3 style="color:var(--text-main); font-size:1.2rem; margin-bottom:15px;">Spending Trend Forecast</h3>', unsafe_allow_html=True)
        
        plot_df = filtered_monthly.copy()
        fig = go.Figure()
        
        mean_spend = plot_df['total_spending'].mean()
        std_spend = plot_df['total_spending'].std()
        spikes = plot_df[plot_df['total_spending'] > mean_spend + 1.5*std_spend]
        
        fig.add_trace(go.Scatter(x=plot_df['year_month'], y=plot_df['total_spending'],
                                 mode='lines+markers', name='Actual Spend',
                                 line=dict(color='#14F1D9', width=3),
                                 marker=dict(size=8, color='#14F1D9')))
                                 
        if 'spending_trend' in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df['year_month'], y=plot_df['spending_trend'],
                                     mode='lines', name='3-Month Avg',
                                     line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dash')))
                                     
        for _, row in spikes.iterrows():
            fig.add_annotation(
                x=row['year_month'], y=row['total_spending'],
                text="Spike Detected",
                showarrow=True, arrowhead=1, arrowcolor="#EF4444",
                bgcolor="#EF4444", font=dict(color="white", size=10),
                ax=-20, ay=-30
            )
                                     
        if len(monthly_summary) >= 4:
            pred_res = predictor.predict_next_month(df)
            if pred_res:
                pred_val, std_dev = pred_res
                last_date = pd.to_datetime(plot_df['year_month'].iloc[-1])
                next_date_str = (last_date + pd.DateOffset(months=1)).strftime('%Y-%m')
                
                fig.add_vline(x=plot_df['year_month'].iloc[-1], line_width=2, line_dash="dash", line_color="#94A3B8")
                fig.add_annotation(
                    x=plot_df['year_month'].iloc[-1], y=mean_spend,
                    text="Forecast Begins", showarrow=False,
                    textangle=-90, xshift=-10, font=dict(color="#94A3B8")
                )
                
                fig.add_trace(go.Scatter(x=[plot_df['year_month'].iloc[-1], next_date_str], 
                                         y=[plot_df['total_spending'].iloc[-1], pred_val],
                                         mode='lines+markers', name='Forecast',
                                         line=dict(color='#3B82F6', width=3, dash='dot'),
                                         marker=dict(size=10, symbol='star', color='#3B82F6')))
                                         
                upper_bound = pred_val + (1.96 * std_dev)
                lower_bound = max(0, pred_val - (1.96 * std_dev))
                
                fig.add_trace(go.Scatter(
                    x=[plot_df['year_month'].iloc[-1], next_date_str, next_date_str, plot_df['year_month'].iloc[-1]],
                    y=[plot_df['total_spending'].iloc[-1], upper_bound, lower_bound, plot_df['total_spending'].iloc[-1]],
                    fill='toself',
                    fillcolor='rgba(59, 130, 246, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence'
                ))
        
        fig.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    with col_v2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:var(--text-main); font-size:1.2rem; margin-bottom:15px; margin-top:0;">⚡ Smart Insights</h3>', unsafe_allow_html=True)
        
        insights = generate_smart_insights(filtered_df, filtered_monthly)
        for insight in insights:
            color = "var(--danger)" if insight['type'] == 'warning' else "var(--success)" if insight['type'] == 'success' else "var(--text-main)"
            st.markdown(f"""
            <div style="margin-bottom: 15px; display:flex; gap:10px; align-items:start;">
                <span style="font-size:1.3rem;">{insight['icon']}</span>
                <span style="color:{color}; font-size:0.95rem; line-height: 1.4;">{insight['text']}</span>
            </div>
            """, unsafe_allow_html=True)
            
        if risk_factors and risk_score > 30:
            st.markdown('<hr style="border-color: rgba(255,255,255,0.1);">', unsafe_allow_html=True)
            st.markdown('<p style="color:var(--text-muted); font-size:0.85rem; font-weight:600; text-transform:uppercase;">Risk Flags Detected</p>', unsafe_allow_html=True)
            for factor in risk_factors:
                st.markdown(f"<p style='color:var(--danger); font-size:0.9rem; margin:0;'>• {factor}</p>", unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: DEEP ANALYSIS ---
elif page == "Deep Analysis":
    st.markdown('<div class="premium-title">Deep Data Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Expense Composition (Luxury vs Essential)")
        recent_month_str = filtered_monthly.iloc[-1]['year_month']
        recent_data = filtered_df[filtered_df['date'].dt.to_period('M').astype(str) == recent_month_str]
        
        if not recent_data.empty:
            fig = px.sunburst(recent_data, path=['necessity', 'category'], values='amount',
                              color='necessity', color_discrete_map={'Essential': '#10B981', 'Luxury': '#F59E0B'},
                              template='plotly_dark')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No data for recent month.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Weekday vs Spending Heatmap")
        
        day_spend = filtered_df.groupby(['weekday', 'category'])['amount'].sum().reset_index()
        cats_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_spend['weekday'] = pd.Categorical(day_spend['weekday'], categories=cats_order, ordered=True)
        day_spend = day_spend.sort_values('weekday')
        
        fig = px.density_heatmap(day_spend, x='weekday', y='category', z='amount',
                                 template='plotly_dark', color_continuous_scale='Teal')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # WHAT CHANGED SECTION
    st.markdown("### What Changed This Month?")
    if len(filtered_monthly) >= 2:
        curr_m = filtered_monthly.iloc[-1]['year_month']
        prev_m = filtered_monthly.iloc[-2]['year_month']
        inc, dec = what_changed_analysis(filtered_df, curr_m, prev_m)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown('<div class="glass-card" style="border-top: 3px solid #EF4444;">', unsafe_allow_html=True)
            st.markdown("#### ⬆️ Major Increases")
            if not inc.empty:
                for _, row in inc.iterrows():
                    st.markdown(f"**{row.name}**: +₹{row['Diff']:,.0f} <span style='color:#EF4444;'>({row['% Change']:.1f}%)</span>", unsafe_allow_html=True)
            else:
                st.write("No major increases.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_c2:
            st.markdown('<div class="glass-card" style="border-top: 3px solid #10B981;">', unsafe_allow_html=True)
            st.markdown("#### ⬇️ Major Savings")
            if not dec.empty:
                for _, row in dec.iterrows():
                    st.markdown(f"**{row.name}**: -₹{abs(row['Diff']):,.0f} <span style='color:#10B981;'>({row['% Change']:.1f}%)</span>", unsafe_allow_html=True)
            else:
                st.write("No major decreases.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: BUDGETS & CONTROL ---
elif page == "Budgets & Control":
    st.markdown('<div class="premium-title">Budget Control Center</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:var(--text-muted);'>Define categorical limits and track real-time adherence to prevent overspending.</p>", unsafe_allow_html=True)
    
    recent_month_str = filtered_monthly.iloc[-1]['year_month']
    recent_data = filtered_df[filtered_df['date'].dt.to_period('M').astype(str) == recent_month_str]
    curr_cat_spend = recent_data.groupby('category')['amount'].sum().reset_index()
    
    # Load budgets
    user_budgets = get_all_budgets(USER_ID)
    cats = df['category'].unique().tolist()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Set Limits")
        with st.form("budget_form"):
            new_budgets = {}
            for cat in cats:
                default_val = float(user_budgets.get(cat, 0.0))
                # Auto-suggest historical average if 0
                if default_val == 0.0:
                    hist_avg = df[df['category'] == cat]['amount'].sum() / len(monthly_summary) if len(monthly_summary) > 0 else 5000
                    default_val = round(float(hist_avg), -2)
                
                val = st.number_input(f"{cat} Limit (₹)", min_value=0.0, value=default_val, step=500.0, key=f"bud_{cat}")
                new_budgets[cat] = val
                
            submitted = st.form_submit_button("Save Budgets")
            if submitted:
                for cat, limit in new_budgets.items():
                    save_budget(USER_ID, cat, limit)
                st.success("Budgets saved securely to SQLite DB!")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Real-Time Budget vs Actual")
        
        bva_list = []
        for index, row in curr_cat_spend.iterrows():
            cat = row['category']
            actual = row['amount']
            budget = user_budgets.get(cat, 0)
            if budget > 0:
                perc = (actual / budget) * 100
                bva_list.append({'Category': cat, 'Actual': actual, 'Budget': budget, '% Used': perc})
                
        if bva_list:
            bva_df = pd.DataFrame(bva_list).sort_values('% Used', ascending=False)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=bva_df['Category'],
                x=bva_df['Budget'],
                name='Budget Limit',
                orientation='h',
                marker=dict(color='rgba(255, 255, 255, 0.1)', line=dict(color='rgba(255, 255, 255, 0.3)', width=1))
            ))
            fig.add_trace(go.Bar(
                y=bva_df['Category'],
                x=bva_df['Actual'],
                name='Actual Spend',
                orientation='h',
                marker=dict(color=np.where(bva_df['% Used'] > 100, '#EF4444', '#14F1D9'))
            ))
            
            fig.update_layout(barmode='overlay', template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Flags
            over_budget = bva_df[bva_df['% Used'] > 100]
            if not over_budget.empty:
                st.markdown('<hr style="border-color: rgba(255,255,255,0.1);">', unsafe_allow_html=True)
                st.markdown('<p style="color:#EF4444; font-weight:600;">⚠️ Overspending Alerts</p>', unsafe_allow_html=True)
                for _, row in over_budget.iterrows():
                    st.markdown(f"**{row['Category']}**: Exceeded budget by ₹{(row['Actual'] - row['Budget']):,.0f} ({row['% Used']:.0f}% used)")
        else:
            st.info("Set your budgets on the left to see the comparison.")
            
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: AI ADVISOR ---
elif page == "AI Advisor & Chat":
    st.markdown('<div class="premium-title">AI Financial Advisor</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color:var(--text-muted); margin-bottom:20px;">
        Chat with your personalized AI to understand your spending, get tips on saving, or ask for a health check.
    </p>
    """, unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Chat input
    if prompt := st.chat_input("Ask about your finances (e.g., 'Am I overspending?')"):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate and add assistant message
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                response = ai_advisor_response(prompt, filtered_df, filtered_monthly)
                st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
    st.markdown("---")
    st.markdown("### Actionable Recommendations")
    recs = generate_recommendations(filtered_df, assumed_income)
    
    cols = st.columns(len(recs) if len(recs) > 0 else 1)
    for idx, rec in enumerate(recs):
        with cols[idx % 3]:
            bg_color = "rgba(239, 68, 68, 0.1)" if rec['type'] == 'warning' else "rgba(16, 185, 129, 0.1)"
            border_color = "var(--danger)" if rec['type'] == 'warning' else "var(--success)"
            st.markdown(f"""
            <div class="glass-card" style="background:{bg_color}; border-left: 4px solid {border_color};">
                <h4 style="margin-top:0;">{rec['icon']} {rec['title']}</h4>
                <p style="font-size:0.9rem; color:var(--text-muted);">{rec['description']}</p>
            </div>
            """, unsafe_allow_html=True)
