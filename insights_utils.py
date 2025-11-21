import pandas as pd
import numpy as np

def generate_detailed_analysis(analysis_df, overall_stability, overall_mttr, overall_mtbf, analysis_level):
    """
    Generates the main automated analysis summary text.
    Expects analysis_df to have columns: 'period', 'stability', 'stops', 'mttr'
    """
    if analysis_df is None or analysis_df.empty:
        return {"error": "Not enough data to generate a trend analysis."}

    # 1. Overall Assessment
    stability_class = "good (above 70%)" if overall_stability > 70 else "needs improvement (50-70%)" if overall_stability > 50 else "poor (below 50%)"
    overall_summary = f"The overall stability for this period is <strong>{overall_stability:.1f}%</strong>, which is considered <strong>{stability_class}</strong>."

    # 2. Predictive Trend
    predictive_insight = ""
    analysis_df_clean = analysis_df.dropna(subset=['stability'])
    
    if len(analysis_df_clean) > 1:
        volatility_std = analysis_df_clean['stability'].std()
        volatility_level = "highly volatile" if volatility_std > 15 else "moderately volatile" if volatility_std > 5 else "relatively stable"
        
        half_point = len(analysis_df_clean) // 2
        first_half_mean = analysis_df_clean['stability'].iloc[:half_point].mean()
        second_half_mean = analysis_df_clean['stability'].iloc[half_point:].mean()
        
        trend_direction = "stable"
        if second_half_mean > first_half_mean * 1.05: trend_direction = "improving"
        elif second_half_mean < first_half_mean * 0.95: trend_direction = "declining"

        if trend_direction == "stable":
            predictive_insight = f"Performance has been <strong>{volatility_level}</strong> with no clear long-term upward or downward trend."
        else:
            predictive_insight = f"Performance shows a <strong>{trend_direction} trend</strong>, although this has been <strong>{volatility_level}</strong>."

    # 3. Best/Worst Performers
    best_worst_analysis = ""
    if not analysis_df_clean.empty:
        best_performer = analysis_df_clean.loc[analysis_df_clean['stability'].idxmax()]
        worst_performer = analysis_df_clean.loc[analysis_df_clean['stability'].idxmin()]

        def format_period_label(val):
            if isinstance(val, (pd.Timestamp, datetime, date)):
                return val.strftime('%Y-%m-%d')
            return str(val)

        best_lbl = format_period_label(best_performer['period'])
        worst_lbl = format_period_label(worst_performer['period'])

        best_worst_analysis = (f"The best performance was during <strong>{best_lbl}</strong> (Stability: {best_performer['stability']:.1f}%), "
                               f"while the worst was during <strong>{worst_lbl}</strong> (Stability: {worst_performer['stability']:.1f}%). "
                               f"The key difference was the impact of stoppages: the worst period had {int(worst_performer['stops'])} stops with an average duration (MTTR) of {worst_performer.get('mttr', 0):.1f} min.")

    # 4. Patterns
    pattern_insight = ""
    if not analysis_df_clean.empty and analysis_df_clean['stops'].sum() > 0:
        if "Daily" in analysis_level and 'period' in analysis_df_clean.columns and isinstance(analysis_df_clean['period'].iloc[0], (int, float)):
             # Likely hourly data
             peak_stop = analysis_df_clean.loc[analysis_df_clean['stops'].idxmax()]
             pattern_insight = f"A notable pattern is the concentration of stop events around <strong>Hour {int(peak_stop['period'])}</strong>."
        else:
            # Check for outliers
            mean_stab = analysis_df_clean['stability'].mean()
            std_stab = analysis_df_clean['stability'].std()
            if std_stab > 0:
                outliers = analysis_df_clean[analysis_df_clean['stability'] < (mean_stab - 1.5 * std_stab)]
                if not outliers.empty:
                    worst_out = outliers.loc[outliers['stability'].idxmin()]
                    pattern_insight = f"A key area of concern is <strong>{format_period_label(worst_out['period'])}</strong>, which performed significantly below average."

    # 5. Recommendation Logic
    recommendation = ""
    if overall_stability >= 95:
        recommendation = "Overall performance is excellent. Continue monitoring for any emerging negative trends."
    elif overall_stability > 70:
        if overall_mtbf > 0 and overall_mttr > 0 and overall_mtbf < (overall_mttr * 5):
            recommendation = f"Performance is good, but could be improved by focusing on <strong>MTBF ({overall_mtbf:.1f} min)</strong>. Investigating frequent minor stops could yield gains."
        else:
            recommendation = f"Performance is good, but could be improved by focusing on <strong>MTTR ({overall_mttr:.1f} min)</strong>. Streamlining the repair process for infrequent stops could yield gains."
    else:
        if overall_mtbf > 0 and overall_mttr > 0 and overall_mtbf < overall_mttr:
            recommendation = f"Stability is poor. The primary driver is low <strong>MTBF ({overall_mtbf:.1f} min)</strong>. Priority: Investigate root causes of frequent breakdowns."
        else:
            recommendation = f"Stability is poor. The primary driver is high <strong>MTTR ({overall_mttr:.1f} min)</strong>. Priority: Investigate why stops take so long to resolve."

    return {
        "overall": overall_summary,
        "predictive": predictive_insight,
        "best_worst": best_worst_analysis,
        "patterns": pattern_insight,
        "recommendation": recommendation
    }

def generate_bucket_analysis(complete_runs, bucket_labels):
    """Generates text analysis for the bucket charts."""
    if complete_runs.empty or 'duration_min' not in complete_runs.columns:
        return "No completed runs to analyze for long-run trends."
        
    total_completed_runs = len(complete_runs)
    
    # Define "Long Run" as >= 60 mins
    long_run_buckets = [label for label in bucket_labels if "60" in label or "+" in label]
    # Simple check if bucket label numeric start >= 60
    def is_long(lbl):
        try:
            val = int(lbl.split(' ')[0].replace('+', ''))
            return val >= 60
        except: return False
        
    long_buckets = [l for l in bucket_labels if is_long(l)]
    
    num_long_runs = complete_runs[complete_runs['time_bucket'].isin(long_buckets)].shape[0] if 'time_bucket' in complete_runs.columns else 0
    percent_long_runs = (num_long_runs / total_completed_runs * 100) if total_completed_runs > 0 else 0
    
    longest_run_min = complete_runs['duration_min'].max()
    
    from shared_utils import format_minutes_to_dhm # Import here to avoid circular dependency at top level
    longest_run_formatted = format_minutes_to_dhm(longest_run_min)
    
    analysis_text = f"Out of <strong>{total_completed_runs}</strong> completed runs, <strong>{num_long_runs}</strong> ({percent_long_runs:.1f}%) qualified as long runs (>60 min). "
    analysis_text += f"The longest single run lasted <strong>{longest_run_formatted}</strong>."
    
    if total_completed_runs > 0:
        if percent_long_runs < 20:
            analysis_text += " Most runs are short, indicating frequent process interruptions."
        elif percent_long_runs > 50:
            analysis_text += " This indicates strong capability for sustained stable operation."
            
    return analysis_text

def generate_mttr_mtbf_analysis(analysis_df, analysis_level):
    """Generates correlation analysis between Stops and Stability."""
    df = analysis_df.dropna(subset=['stops', 'stability', 'mttr'])
    if len(df) < 2 or df['stops'].sum() == 0:
        return "Not enough data for correlation analysis."
        
    stops_corr = df['stops'].corr(df['stability'])
    mttr_corr = df['mttr'].corr(df['stability'])
    
    driver = "unknown"
    if abs(stops_corr) > abs(mttr_corr):
        driver = "frequency of stops (MTBF issues)"
    else:
        driver = "duration of stops (MTTR issues)"
        
    text = f"Statistical correlation suggests stability is most impacted by the <strong>{driver}</strong>."
    return text