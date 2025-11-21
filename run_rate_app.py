import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import warnings
import streamlit.components.v1 as components
from datetime import datetime, timedelta, date

# --- IMPORT FROM SHARED UTILS ---
from shared_utils import (
    ProductionMetricsCalculator,
    calculate_run_summaries,
    calculate_daily_summaries,
    calculate_weekly_summaries,
    load_data_unified,
    load_all_data_unified, 
    format_seconds_to_dhm,
    format_duration,
    PASTEL_COLORS,
    calculate_capacity_risk_summary
)

# --- IMPORT FROM INSIGHTS UTILS ---
from insights_utils import (
    generate_detailed_analysis,
    generate_bucket_analysis,
    generate_mttr_mtbf_analysis
)

# ==============================================================================
# --- 1. PAGE CONFIG & PLOTTING FUNCTIONS ---
# ==============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(layout="wide", page_title="Run Rate Analysis Dashboard")

def create_gauge(value, title, steps=None):
    gauge_config = {'axis': {'range': [0, 100]}}
    if steps:
        gauge_config['steps'] = steps; gauge_config['bar'] = {'color': '#262730'}
    else:
        gauge_config['bar'] = {'color': "darkblue"}; gauge_config['bgcolor'] = "lightgray"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, title={'text': title}, gauge=gauge_config))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_shot_bar_chart(df, lower_limit, upper_limit, mode_ct):
    if df.empty:
        st.info("No shot data to display."); return
    df = df.copy()
    df['color'] = np.where(df['stop_flag'] == 1, PASTEL_COLORS['red'], '#3498DB')
    
    downtime_gap_indices = df[df['adj_ct_sec'] != df['Actual CT']].index
    valid_downtime_gap_indices = downtime_gap_indices[downtime_gap_indices > 0]
    normal_shot_indices = df.index.difference(valid_downtime_gap_indices)

    if not normal_shot_indices.empty:
        shot_index_in_second = df.loc[normal_shot_indices].groupby('shot_time').cumcount()
        time_offset = pd.to_timedelta(shot_index_in_second * 0.2, unit='s')
        df.loc[normal_shot_indices, 'plot_time'] = df.loc[normal_shot_indices, 'shot_time'] + time_offset
    
    if not valid_downtime_gap_indices.empty:
        prev_shot_timestamps = df['shot_time'].shift(1).loc[valid_downtime_gap_indices]
        df.loc[valid_downtime_gap_indices, 'plot_time'] = prev_shot_timestamps

    if 0 in df.index and pd.isna(df.loc[0, 'plot_time']):
        df.loc[0, 'plot_time'] = df.loc[0, 'shot_time']
         
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['plot_time'], y=df['adj_ct_sec'], marker_color=df['color'], name='Cycle Time', showlegend=False))
    fig.add_trace(go.Bar(x=[None], y=[None], name="Normal Shot", marker_color='#3498DB', showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], name="Stopped Shot", marker_color=PASTEL_COLORS['red'], showlegend=True))
    
    if not df.empty:
        fig.add_shape(type="rect", xref="x", yref="y", x0=df['shot_time'].min(), y0=lower_limit, x1=df['shot_time'].max(), y1=upper_limit, fillcolor=PASTEL_COLORS['green'], opacity=0.3, layer="below", line_width=0)
            
    y_axis_cap = min(max(mode_ct * 2, 50), 500)
    fig.update_layout(title="Cycle Time Analysis", xaxis_title="Time", yaxis_title="Cycle Time (sec)", yaxis=dict(range=[0, y_axis_cap]), bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

def plot_trend_chart(df, x_col, y_col, title, y_title):
    if df is None or df.empty or y_col not in df.columns: return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="lines+markers", name=y_title))
    fig.update_layout(title=title, yaxis_title=y_title)
    st.plotly_chart(fig, use_container_width=True)

def plot_mttr_mtbf_chart(df, x_col):
    if df is None or df.empty: return
    # Check if columns exist, if not try to fallback to known aliases or return
    mttr_col = 'mttr_min' if 'mttr_min' in df.columns else 'mttr'
    mtbf_col = 'mtbf_min' if 'mtbf_min' in df.columns else 'mtbf'
    
    if mttr_col not in df.columns or mtbf_col not in df.columns:
        st.warning(f"Missing MTTR/MTBF columns for chart. Available: {df.columns.tolist()}")
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df[x_col], y=df[mttr_col], name='MTTR (min)', line=dict(color='red')), secondary_y=False)
    fig.add_trace(go.Scatter(x=df[x_col], y=df[mtbf_col], name='MTBF (min)', line=dict(color='green')), secondary_y=True)
    fig.update_layout(title="MTTR & MTBF Trend")
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# --- 2. RISK TOWER (UNIFIED) ---
# ==============================================================================

@st.cache_data(show_spinner="Analyzing Risk...")
def calculate_risk_scores_cached(df_all_tools):
    id_col = "tool_id"
    initial_metrics = []
    RUN_INTERVAL_HOURS = 8.0
    
    for tool_id, df_tool in df_all_tools.groupby(id_col):
        if df_tool.empty or len(df_tool) < 10: continue
        
        calc = ProductionMetricsCalculator(df_tool, 0.05, 2.0, RUN_INTERVAL_HOURS)
        df_proc = calc.results['processed_df']
        
        end_date = df_proc['shot_time'].max()
        start_date = end_date - timedelta(weeks=4)
        df_period = df_proc[(df_proc['shot_time'] >= start_date) & (df_proc['shot_time'] <= end_date)].copy()
        if df_period.empty: continue
        
        is_new_run = df_period['time_diff_sec'] > (RUN_INTERVAL_HOURS * 3600)
        df_period['run_label'] = is_new_run.cumsum().apply(lambda x: f"Run_{x}")
        
        run_sums = calculate_run_summaries(df_period, 0.05, 2.0, RUN_INTERVAL_HOURS)
        if run_sums.empty: continue
        
        tot_run = run_sums['total_runtime_sec'].sum()
        prod_time = run_sums['production_time_sec'].sum()
        # Use the correct key from run_sums
        down_time = run_sums['downtime_sec'].sum()
        stops = run_sums['stops'].sum()
        
        stability = (prod_time / tot_run * 100) if tot_run > 0 else 100.0
        mttr = (down_time / 60 / stops) if stops > 0 else 0
        
        initial_metrics.append({
            'Tool ID': tool_id,
            'Stability': stability,
            'MTTR': mttr,
            'Analysis Period': f"{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}"
        })
        
    if not initial_metrics: return pd.DataFrame()
    df_metrics = pd.DataFrame(initial_metrics)
    df_metrics['Risk Score'] = df_metrics['Stability'] 
    return df_metrics.sort_values('Risk Score')

def render_risk_tower(df_all_tools):
    st.title("Risk Tower (Unified)")
    risk_df = calculate_risk_scores_cached(df_all_tools)
    if not risk_df.empty:
        def style_risk(row):
            score = row['Risk Score']
            color = PASTEL_COLORS['green'] if score > 70 else PASTEL_COLORS['orange'] if score > 50 else PASTEL_COLORS['red']
            return [f'background-color: {color}' for _ in row]
        st.dataframe(risk_df.style.apply(style_risk, axis=1).format({'Stability': '{:.1f}%', 'MTTR': '{:.1f}m'}), use_container_width=True)
    else:
        st.warning("Not enough data for Risk Tower.")

# ==============================================================================
# --- 3. MAIN DASHBOARD ---
# ==============================================================================

def render_dashboard(df_tool, tool_id_selection):
    st.sidebar.title("Dashboard Controls")
    
    analysis_level = st.sidebar.radio("Analysis Level", ["Daily", "Weekly (by Run)", "Monthly (by Run)", "Custom Period (by Run)"])
    
    st.sidebar.markdown("---")
    tolerance = st.sidebar.slider("Tolerance Band (% of Mode CT)", 0.01, 0.50, 0.25, 0.01)
    downtime_gap = st.sidebar.slider("Downtime Gap (sec)", 0.0, 5.0, 2.0, 0.5)
    run_interval = st.sidebar.slider("Run Interval Threshold (hours)", 1, 24, 8, 1)
    
    st.sidebar.markdown("---")
    detailed_view = st.sidebar.toggle("Show Detailed Analysis", value=True)
    
    @st.cache_data
    def get_dates(df):
        df['date'] = df['shot_time'].dt.date
        df['week'] = df['shot_time'].dt.isocalendar().week
        df['month'] = df['shot_time'].dt.to_period('M')
        return df
    df_dates = get_dates(df_tool.copy())
    
    df_view = pd.DataFrame()
    sub_header = ""
    
    if "Daily" in analysis_level:
        dates = sorted(df_dates['date'].unique())
        sel_date = st.selectbox("Select Date", dates, index=len(dates)-1)
        df_view = df_dates[df_dates['date'] == sel_date]
        sub_header = f"Summary for {sel_date}"
    elif "Weekly" in analysis_level:
        weeks = sorted(df_dates['week'].unique())
        sel_week = st.selectbox("Select Week", weeks, index=len(weeks)-1)
        df_view = df_dates[df_dates['week'] == sel_week]
        sub_header = f"Summary for Week {sel_week}"
    elif "Monthly" in analysis_level:
        months = sorted(df_dates['month'].unique())
        sel_month = st.selectbox("Select Month", months, index=len(months)-1)
        df_view = df_dates[df_dates['month'] == sel_month]
        sub_header = f"Summary for {sel_month}"
    elif "Custom" in analysis_level:
        d_min, d_max = df_dates['date'].min(), df_dates['date'].max()
        s_date = st.date_input("Start", d_min); e_date = st.date_input("End", d_max)
        df_view = df_dates[(df_dates['date'] >= s_date) & (df_dates['date'] <= e_date)]
        sub_header = f"Summary for {s_date} to {e_date}"

    if df_view.empty:
        st.warning("No data."); return

    # --- MAIN CALCULATION ---
    st.title(f"Run Rate: {tool_id_selection}")
    st.subheader(sub_header)
    
    if 'by Run' in analysis_level:
        is_new_run = df_view['shot_time'].diff().dt.total_seconds() > (run_interval * 3600)
        df_view['run_label'] = is_new_run.cumsum().apply(lambda x: f"Run_{x+1}")
        
        run_summaries = calculate_run_summaries(df_view, tolerance, downtime_gap, run_interval)
        
        total_runtime = run_summaries['total_runtime_sec'].sum()
        prod_time = run_summaries['production_time_sec'].sum()
        downtime = run_summaries['downtime_sec'].sum()
        stops = run_summaries['stops'].sum()
        total_shots = run_summaries['total_shots'].sum()
        
        stability = (prod_time / total_runtime * 100) if total_runtime > 0 else 0
        mttr = (downtime / 60 / stops) if stops > 0 else 0
        mtbf = (prod_time / 60 / stops) if stops > 0 else (prod_time / 60)
        
        calc_full = ProductionMetricsCalculator(df_view, tolerance, downtime_gap, run_interval)
        res_full = calc_full.results
        
        # Keep original column names for plotting, or rename carefully
        trend_df = run_summaries.rename(columns={'run_label': 'period'})
        # Ensure we have mttr_min/mtbf_min from calculate_run_summaries
        trend_x = 'period'
        
    else:
        calc = ProductionMetricsCalculator(df_view, tolerance, downtime_gap, run_interval)
        res = calc.results
        res_full = res
        
        total_runtime = res['total_runtime_sec']
        prod_time = res['production_time_sec']
        downtime = res['downtime_sec']
        stops = res['stop_events']
        total_shots = res['total_shots']
        stability = res['stability_index']
        mttr = res['mttr_min']
        mtbf = res['mtbf_min']
        
        # Ensure hourly summary has 'period'
        trend_df = res['hourly_summary'].copy()
        trend_df['period'] = trend_df['hour']
        # It already has 'mttr_min' and 'mtbf_min' from shared_utils
        trend_x = 'period'

    # --- METRICS DISPLAY ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Runtime", format_duration(total_runtime))
    c2.metric("Production Time", format_duration(prod_time))
    c3.metric("Run Rate Downtime", format_duration(downtime))
    c4.metric("Stops", stops)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("MTTR", format_duration(mttr * 60))
    c2.metric("MTBF", format_duration(mtbf * 60))
    c3.metric("Stability", f"{stability:.1f}%")

    st.divider()
    
    # --- CHARTS ---
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cycle Time Visualization")
        plot_shot_bar_chart(res_full['processed_df'], res_full['lower_limit'], res_full['upper_limit'], res_full['mode_ct'])
    with c2:
        st.subheader("Stability Trend")
        if not trend_df.empty:
            # Use correct column name for stability
            stab_col = 'stability_index' if 'stability_index' in trend_df.columns else 'stability'
            plot_trend_chart(trend_df, trend_x, stab_col, "Stability Trend", "Stability %")

    if not trend_df.empty:
        st.subheader("MTTR & MTBF Trend")
        plot_mttr_mtbf_chart(trend_df, trend_x)

    # --- AI INSIGHTS ---
    if detailed_view:
        st.markdown("---")
        with st.expander("🤖 Automated Analysis Summary", expanded=False):
            # Prepare DF for insights (needs specific column names)
            insight_df = trend_df.copy()
            
            # Map columns to expected names for insights
            rename_map = {
                'stability_index': 'stability', 
                'mttr_min': 'mttr', 
                'mtbf_min': 'mtbf',
                'stops': 'stops'
                # 'period' is already there from above logic
            }
            insight_df.rename(columns=rename_map, inplace=True)
            
            insights = generate_detailed_analysis(insight_df, stability, mttr, mtbf, analysis_level)
            mttr_insight = generate_mttr_mtbf_analysis(insight_df, analysis_level)
            
            if "error" in insights:
                st.error(insights["error"])
            else:
                st.markdown(f"""
                ### Overall Assessment
                {insights['overall']}
                
                ### Predictive Trend
                {insights['predictive']}
                
                ### Performance Variance
                {insights['best_worst']}
                
                ### Key Recommendation
                {insights['recommendation']}
                
                ---
                ### Correlation Analysis
                {mttr_insight}
                """, unsafe_allow_html=True)

    with st.expander("View Detailed Data"):
        st.dataframe(res_full['processed_df'])

# ==============================================================================
# --- 4. UPLOAD & LAUNCH ---
# ==============================================================================

st.sidebar.title("Upload")
files = st.sidebar.file_uploader("Excel Files", accept_multiple_files=True)

if files:
    df_all = load_all_data_unified(files)
    if not df_all.empty:
        tools = ["Risk Tower"] + sorted(df_all['tool_id'].unique().tolist())
        sel = st.sidebar.selectbox("Select Tool", tools)
        
        t1, t2 = st.tabs(["Risk Tower", "Dashboard"])
        with t1: render_risk_tower(df_all)
        with t2:
            if sel != "Risk Tower":
                render_dashboard(df_all[df_all['tool_id'] == sel], sel)
            else:
                st.info("Select a tool.")
    else:
        st.error("No data found.")
else:
    st.info("Upload files.")