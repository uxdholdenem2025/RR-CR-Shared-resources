import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

# ==============================================================================
# --- 1. SHARED CONSTANTS & FORMATTERS ---
# ==============================================================================

PASTEL_COLORS = {
    'red': '#ff6961',
    'orange': '#ffb347',
    'green': '#77dd77',
    'blue': '#3498DB'
}

# Unified Column Names for Reporting
ALL_RESULT_COLUMNS = [
    'Date', 
    'Total Run Time (sec)', 
    'Optimal Output (parts)',
    'Run Rate Downtime (Stops) (sec)',
    'Run Rate Downtime (Stops) (parts)',
    'Actual Output (parts)', 
    'Actual Cycle Time Total (sec)',
    'Cycle Time Efficiency Gain (Fast Cycles) (sec)',   
    'Cycle Time Efficiency Loss (Slow Cycles) (sec)',   
    'Cycle Time Efficiency Loss (Slow Cycles) (parts)', 
    'Cycle Time Efficiency Gain (Fast Cycles) (parts)', 
    'Total Capacity Loss (parts)', 
    'Total Capacity Loss (sec)',
    'Target Output (parts)', 
    'Gap to Target (parts)',
    'Capacity Loss (vs Target) (parts)', 
    'Capacity Loss (vs Target) (sec)',
    'Total Shots (all)', 
    'Production Shots', 
    'Downtime Shots',
    'Run Rate Efficiency (%)' 
]

def format_seconds_to_dhm(total_seconds):
    """Converts total seconds into a 'Xd Yh Zm' string."""
    if pd.isna(total_seconds) or total_seconds < 0: return "N/A"
    
    if total_seconds < 60:
        return f"{total_seconds:.1f}s"

    total_minutes = int(total_seconds / 60)
    days = total_minutes // (60 * 24)
    remaining_minutes = total_minutes % (60 * 24)
    hours = remaining_minutes // 60
    minutes = remaining_minutes % 60
    
    parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0 or not parts: parts.append(f"{minutes}m")
    
    return " ".join(parts) if parts else "0m"

def format_duration(seconds):
    return format_seconds_to_dhm(seconds)

# ==============================================================================
# --- 2. DATA LOADING (UNIFIED) ---
# ==============================================================================

@st.cache_data
def load_data_unified(uploaded_file):
    """Loads data from a SINGLE uploaded file (Excel or CSV) into a DataFrame."""
    try:
        # Handle both Streamlit UploadedFile objects and normal file paths/buffers
        if hasattr(uploaded_file, 'seek'):
             uploaded_file.seek(0)
        
        if hasattr(uploaded_file, 'name') and uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            # Default to excel for xlsx or unknown
            df = pd.read_excel(uploaded_file)
            
        df.columns = df.columns.str.strip()
        
        col_map = {
            'TOOLING ID': 'tool_id', 'EQUIPMENT CODE': 'tool_id', 'Tool ID': 'tool_id',
            'SHOT TIME': 'shot_time', 'Time': 'shot_time', 'timestamp': 'shot_time',
            'ACTUAL CT': 'Actual CT', 'Actual Cycle Time': 'Actual CT', 'actual ct': 'Actual CT',
            'APPROVED CT': 'Approved CT', 'Approved Cycle Time': 'Approved CT', 'approved ct': 'Approved CT',
            'Working Cavities': 'Working Cavities', 'working cavities': 'Working Cavities'
        }
        new_cols = {}
        for col in df.columns:
            for key, val in col_map.items():
                if col.lower() == key.lower():
                    new_cols[col] = val
                    break
        df.rename(columns=new_cols, inplace=True)
        
        if 'shot_time' not in df.columns:
            if {"YEAR", "MONTH", "DAY", "TIME"}.issubset(df.columns):
                datetime_str = df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-" + df["DAY"].astype(str) + " " + df['TIME'].astype(str)
                df["shot_time"] = pd.to_datetime(datetime_str, errors="coerce")
        else:
            df['shot_time'] = pd.to_datetime(df['shot_time'], errors='coerce')

        df.dropna(subset=['shot_time'], inplace=True)
        
        for col in ['Actual CT', 'Approved CT', 'Working Cavities']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # If tool_id is missing, try to use filename or default
        if 'tool_id' not in df.columns:
             if hasattr(uploaded_file, 'name'):
                 df['tool_id'] = uploaded_file.name.split('.')[0]
             else:
                 df['tool_id'] = 'Unknown_Tool'

        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

@st.cache_data
def load_all_data_unified(files):
    """Loads and combines MULTIPLE uploaded Excel/CSV files."""
    df_list = []
    for file in files:
        try:
            # Call the single file loader
            df = load_data_unified(file)
            
            if not df.empty:
                # Ensure tool_id exists for the combined dataframe logic
                if "tool_id" not in df.columns:
                     df['tool_id'] = file.name.split('.')[0]
                
                # Ensure tool_id is string
                df['tool_id'] = df['tool_id'].astype(str)
                df_list.append(df)
                
        except Exception as e:
            st.warning(f"Could not load file: {file.name}. Error: {e}")
    
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)

# ==============================================================================
# --- 3. CORE CALCULATION ENGINE (THE "SINGLE TRUTH") ---
# ==============================================================================

class ProductionMetricsCalculator:
    """
    The Unified Calculation Engine.
    Calculates Stop Flags, Adjusted Cycle Times, and Aggregates.
    """
    def __init__(self, df: pd.DataFrame, mode_tolerance: float, gap_tolerance: float, run_interval_hours=8.0):
        self.df = df.copy()
        self.mode_tolerance = mode_tolerance
        self.gap_tolerance = gap_tolerance
        self.run_interval_seconds = run_interval_hours * 3600
        self.results = self._calculate()

    def _calculate(self) -> dict:
        df = self.df
        if df.empty or 'Actual CT' not in df.columns:
            return {}

        # 1. Sort & Time Diff
        df = df.sort_values('shot_time').reset_index(drop=True)
        df['time_diff_sec'] = df['shot_time'].diff().dt.total_seconds().fillna(0)
        if len(df) > 0: df.loc[0, 'time_diff_sec'] = df.loc[0, 'Actual CT']

        # 2. Calculate Mode & Limits
        valid_cycles = df[df['Actual CT'] < 999.9]['Actual CT']
        if not valid_cycles.empty:
            mode_ct = valid_cycles.mode().iloc[0] if not valid_cycles.mode().empty else valid_cycles.mean()
        else:
            mode_ct = df['Actual CT'].mean() if not df.empty else 0

        lower_limit = mode_ct * (1 - self.mode_tolerance)
        upper_limit = mode_ct * (1 + self.mode_tolerance)

        # 3. Flagging Logic (Unified)
        is_hard_stop = df['Actual CT'] >= 999.9
        prev_ct = df['Actual CT'].shift(1).fillna(mode_ct)
        is_time_gap = df['time_diff_sec'] > (prev_ct + self.gap_tolerance)
        is_abnormal = ((df['Actual CT'] < lower_limit) | (df['Actual CT'] > upper_limit)) & ~is_hard_stop
        is_run_break = df['time_diff_sec'] > self.run_interval_seconds

        df['stop_flag'] = np.where(is_hard_stop | is_time_gap | is_abnormal, 1, 0)
        if not df.empty: df.loc[0, 'stop_flag'] = 0
        df['stop_event'] = (df['stop_flag'] == 1) & (df['stop_flag'].shift(1, fill_value=0) == 0)

        # 4. Adjusted Cycle Time (adj_ct_sec)
        df['adj_ct_sec'] = df['Actual CT']
        df.loc[is_hard_stop, 'adj_ct_sec'] = 0
        df.loc[is_time_gap, 'adj_ct_sec'] = df['time_diff_sec']
        df.loc[is_run_break, 'adj_ct_sec'] = 0

        # 5. Capacity Risk Specific Logic (Fast/Slow/Target)
        if 'Approved CT' in df.columns:
            ref_ct = df['Approved CT'].mean()
            if pd.isna(ref_ct) or ref_ct == 0: ref_ct = mode_ct
        else:
            ref_ct = mode_ct
            
        df['reference_ct'] = ref_ct
        is_slow = (df['stop_flag'] == 0) & (df['Actual CT'] > ref_ct) & ~np.isclose(df['Actual CT'], ref_ct)
        is_fast = (df['stop_flag'] == 0) & (df['Actual CT'] < ref_ct) & ~np.isclose(df['Actual CT'], ref_ct)
        cavities = df['Working Cavities'] if 'Working Cavities' in df.columns else 1
        
        df['time_loss_sec'] = np.where(is_slow, df['Actual CT'] - ref_ct, 0)
        df['time_gain_sec'] = np.where(is_fast, ref_ct - df['Actual CT'], 0)
        df['parts_loss'] = np.where(is_slow, ((df['Actual CT'] - ref_ct)/ref_ct) * cavities, 0)
        df['parts_gain'] = np.where(is_fast, ((ref_ct - df['Actual CT'])/ref_ct) * cavities, 0)

        # 6. Aggregations (Run Rate)
        total_shots = len(df)
        stop_event_count = df['stop_event'].sum()
        production_time_sec = df.loc[df['stop_flag'] == 0, 'Actual CT'].sum()
        
        # Total Runtime (Timestamp Method)
        if total_shots > 1:
            raw_duration = (df['shot_time'].iloc[-1] - df['shot_time'].iloc[0]).total_seconds() + df['Actual CT'].iloc[-1]
            excluded_duration = df.loc[is_run_break, 'time_diff_sec'].sum()
            total_runtime_sec = raw_duration - excluded_duration
        else:
            total_runtime_sec = df['Actual CT'].iloc[0] if total_shots == 1 else 0
            
        downtime_sec = max(0, total_runtime_sec - production_time_sec)

        # KPIs
        mttr_min = (downtime_sec / 60 / stop_event_count) if stop_event_count > 0 else 0
        mtbf_min = (production_time_sec / 60 / stop_event_count) if stop_event_count > 0 else (production_time_sec / 60)
        stability = (production_time_sec / total_runtime_sec * 100) if total_runtime_sec > 0 else 0.0
        
        # Run Rate Efficiency (Yield)
        normal_shots = len(df[df['stop_flag'] == 0])
        rr_efficiency = (normal_shots / total_shots * 100) if total_shots > 0 else 0.0
        
        df['hour'] = df['shot_time'].dt.hour
        hourly_summary = self._generate_hourly_summary(df)

        return {
            'processed_df': df, 'mode_ct': mode_ct,
            'lower_limit': lower_limit, 'upper_limit': upper_limit,
            'total_shots': total_shots, 'stop_events': stop_event_count,
            'production_time_sec': production_time_sec, 'downtime_sec': downtime_sec,
            'total_runtime_sec': total_runtime_sec,
            'mttr_min': mttr_min, 'mtbf_min': mtbf_min, 'stability_index': stability,
            'rr_efficiency': rr_efficiency,
            'hourly_summary': hourly_summary.fillna(0)
        }

    def _generate_hourly_summary(self, df):
        hourly_groups = df.groupby('hour')
        stops = hourly_groups['stop_event'].sum()
        shots = hourly_groups.size().rename('total_shots')
        uptime_min = df[df['stop_flag'] == 0].groupby('hour')['Actual CT'].sum() / 60
        hourly_down_sec = hourly_groups.apply(lambda x: x[x['stop_flag'] == 1]['adj_ct_sec'].sum())
        
        hourly_summary = pd.DataFrame(index=range(24))
        hourly_summary['hour'] = hourly_summary.index
        hourly_summary = hourly_summary.join(stops.rename('stops')).join(shots).join(uptime_min.rename('uptime_min')).fillna(0)
        hourly_summary = hourly_summary.join(hourly_down_sec.rename('total_downtime_sec')).fillna(0)
        
        hourly_summary['mttr_min'] = (hourly_summary['total_downtime_sec'] / 60) / hourly_summary['stops'].replace(0, np.nan)
        hourly_summary['mtbf_min'] = hourly_summary['uptime_min'] / hourly_summary['stops'].replace(0, np.nan)
        hourly_summary['mtbf_min'] = hourly_summary['mtbf_min'].fillna(hourly_summary['uptime_min'])
        
        eff_run_min = hourly_summary['uptime_min'] + (hourly_summary['total_downtime_sec'] / 60)
        hourly_summary['stability_index'] = np.where(
            eff_run_min > 0, (hourly_summary['uptime_min'] / eff_run_min) * 100, np.where(hourly_summary['stops'] == 0, 100.0, 0.0)
        )
        hourly_summary['stability_index'] = np.where(hourly_summary['total_shots'] == 0, np.nan, hourly_summary['stability_index'])
        return hourly_summary

# ==============================================================================
# --- 4. HELPERS FOR APPS ---
# ==============================================================================

def calculate_capacity_risk_summary(calc_result, target_perc=90.0):
    """Specific aggregator for Capacity Risk App."""
    df = calc_result['processed_df']
    res = {}
    
    cavities = df['Working Cavities'].max() if 'Working Cavities' in df.columns else 1
    if pd.isna(cavities) or cavities == 0: cavities = 1
    ref_ct = df['reference_ct'].mean()
    
    # Unified Mapping
    res['Total Run Time (sec)'] = calc_result['total_runtime_sec']
    res['Optimal Output (parts)'] = (calc_result['total_runtime_sec'] / ref_ct) * cavities if ref_ct > 0 else 0
    res['Actual Output (parts)'] = df.loc[df['stop_flag']==0, 'Working Cavities'].sum()
    res['Run Rate Downtime (Stops) (sec)'] = calc_result['downtime_sec']
    
    # Updated Keys for Cycle Time Efficiency
    res['Cycle Time Efficiency Loss (Slow Cycles) (sec)'] = df['time_loss_sec'].sum()
    res['Cycle Time Efficiency Gain (Fast Cycles) (sec)'] = df['time_gain_sec'].sum()
    res['Cycle Time Efficiency Loss (Slow Cycles) (parts)'] = df['parts_loss'].sum()
    res['Cycle Time Efficiency Gain (Fast Cycles) (parts)'] = df['parts_gain'].sum()
    
    net_cycle_loss_parts = res['Cycle Time Efficiency Loss (Slow Cycles) (parts)'] - res['Cycle Time Efficiency Gain (Fast Cycles) (parts)']
    res['Run Rate Downtime (Stops) (parts)'] = res['Optimal Output (parts)'] - res['Actual Output (parts)'] - net_cycle_loss_parts
    
    target_ratio = target_perc / 100.0
    res['Target Output (parts)'] = res['Optimal Output (parts)'] * target_ratio
    res['Gap to Target (parts)'] = res['Actual Output (parts)'] - res['Target Output (parts)']
    res['Capacity Loss (vs Target) (parts)'] = max(0, res['Target Output (parts)'] - res['Actual Output (parts)'])
    res['Total Capacity Loss (parts)'] = res['Run Rate Downtime (Stops) (parts)'] + net_cycle_loss_parts
    
    res['Total Shots (all)'] = calc_result['total_shots']
    res['Production Shots'] = len(df[df['stop_flag']==0])
    res['Downtime Shots'] = len(df[df['stop_flag']==1])
    res['Run Rate Efficiency (%)'] = calc_result['rr_efficiency']
    
    return res

def calculate_run_summaries(df_period, tolerance, gap_tolerance, run_interval_hours):
    """For Run Rate App: Calculates metrics per run."""
    run_summary_list = []
    group_col = 'run_label' if 'run_label' in df_period.columns else 'run_id'
    
    for label, df_run in df_period.groupby(group_col):
        if not df_run.empty:
            calc = ProductionMetricsCalculator(df_run, tolerance, gap_tolerance, run_interval_hours)
            res = calc.results
            run_summary_list.append({
                'run_label': label,
                'start_time': df_run['shot_time'].min(),
                'end_time': df_run['shot_time'].max(),
                'total_shots': res['total_shots'],
                'normal_shots': len(df_run[df_run['stop_flag'] == 0]),
                'stopped_shots': len(df_run[df_run['stop_flag'] == 1]),
                'mode_ct': res['mode_ct'],
                'lower_limit': res['lower_limit'],
                'upper_limit': res['upper_limit'],
                'total_runtime_sec': res['total_runtime_sec'],
                'production_time_sec': res['production_time_sec'],
                'downtime_sec': res['downtime_sec'],
                'mttr_min': res['mttr_min'],
                'mtbf_min': res['mtbf_min'],
                'stability_index': res['stability_index'],
                'rr_efficiency': res['rr_efficiency'],
                'stops': res['stop_events']
            })
    if not run_summary_list: return pd.DataFrame()
    return pd.DataFrame(run_summary_list).sort_values('start_time').reset_index(drop=True)

def calculate_daily_summaries(df_week, tolerance, gap_tolerance, run_interval_hours):
    results_list = []
    for date_val in sorted(df_week['shot_time'].dt.date.unique()):
        df_day = df_week[df_week['shot_time'].dt.date == date_val]
        if not df_day.empty:
            calc = ProductionMetricsCalculator(df_day, tolerance, gap_tolerance, run_interval_hours)
            res = calc.results
            results_list.append({
                'date': date_val,
                'stability_index': res['stability_index'],
                'mttr_min': res['mttr_min'],
                'mtbf_min': res['mtbf_min'],
                'stops': res['stop_events'],
                'total_shots': res['total_shots'],
                'rr_efficiency': res['rr_efficiency']
            })
    return pd.DataFrame(results_list)

def calculate_weekly_summaries(df_month, tolerance, gap_tolerance, run_interval_hours):
    results_list = []
    df_month['week'] = df_month['shot_time'].dt.isocalendar().week
    for week_val in sorted(df_month['week'].unique()):
        df_week = df_month[df_month['week'] == week_val]
        if not df_week.empty:
            calc = ProductionMetricsCalculator(df_week, tolerance, gap_tolerance, run_interval_hours)
            res = calc.results
            results_list.append({
                'week': week_val,
                'stability_index': res['stability_index'],
                'mttr_min': res['mttr_min'],
                'mtbf_min': res['mtbf_min'],
                'stops': res['stop_events'],
                'total_shots': res['total_shots'],
                'rr_efficiency': res['rr_efficiency']
            })
    return pd.DataFrame(results_list)