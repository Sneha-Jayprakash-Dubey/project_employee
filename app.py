import json

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from flask import g
import pandas as pd
import numpy as np
import joblib
import os
from functools import wraps

from utils.database import (
    save_prediction,
    get_history,
    get_dashboard_stats,
    get_department_averages,
    get_risk_counts,
    get_recent_predictions,
    get_unique_employee_count,
    get_last_prediction_datetime,
    get_department_goals,
    upsert_department_goal,
    get_employee_summary,
    log_activity,
    get_activity_log,
)
from utils.burnout_model import burnout_risk
from utils.shap_explainer import explain_prediction
from utils.analytics import department_productivity, burnout_alerts
from utils.pdf_report import generate_pdf
from model.train_model import train_and_save_model

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "hr_ai_secret")

ADMIN_USER = os.environ.get("ADMIN_USER", "hr")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "hr123")

# Harden session cookies for production.
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
if os.environ.get("FLASK_ENV") == "production":
    app.config["SESSION_COOKIE_SECURE"] = True


# -------------------------------
# Load ML Assets (train if missing)
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_VERSION = 2


def fig_to_json_obj(fig):
    """Convert Plotly figure to JSON-safe dict (handles numpy arrays)."""
    import plotly

    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))


def normalize_filter(value):
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def normalize_risk_filter(value):
    value = normalize_filter(value)
    if value in ("High", "Moderate", "Low"):
        return value
    return None


def chart_title_from_json(chart_json, fallback):
    if not isinstance(chart_json, dict):
        return fallback
    layout = chart_json.get("layout", {}) or {}
    title = layout.get("title")
    if isinstance(title, dict):
        title = title.get("text")
    return title or fallback


def build_analytics_query(
    days=None,
    employee_id=None,
    department=None,
    risk=None,
    team_size_band=None,
    experience_band=None,
    salary_band=None,
    remote_band=None,
):
    query = {}
    if days not in (None, ""):
        query["days"] = days
    if employee_id:
        query["employee_id"] = employee_id
    if department:
        query["department"] = department
    if risk:
        query["risk"] = risk
    if team_size_band:
        query["team_size_band"] = team_size_band
    if experience_band:
        query["experience_band"] = experience_band
    if salary_band:
        query["salary_band"] = salary_band
    if remote_band:
        query["remote_band"] = remote_band
    return query


def normalize_band(value, allowed):
    value = normalize_filter(value)
    if value in allowed:
        return value
    return None


def load_baseline_feature_values():
    """Build realistic default values from training data for sparse forms."""
    defaults = {}
    data_path = os.path.join(BASE_DIR, "dataset", "employee_data.csv")
    if not os.path.exists(data_path):
        return defaults

    try:
        df = pd.read_csv(data_path)

        numeric_defaults = {
            "Age": 35,
            "Years_At_Company": 5,
            "Monthly_Salary": 50,  # model uses salary in thousands
            "Work_Hours_Per_Week": 40,
            "Projects_Handled": 5,
            "Overtime_Hours": 4,
            "Sick_Days": 2,
            "Remote_Work_Frequency": 2,
            "Team_Size": 8,
            "Training_Hours": 10,
            "Promotions": 1,
            "Employee_Satisfaction_Score": 70,
        }

        for key in list(numeric_defaults.keys()):
            if key in df.columns and df[key].notna().any():
                value = float(df[key].median())
                if key == "Monthly_Salary":
                    value = value / 1000.0
                numeric_defaults[key] = value
        defaults.update(numeric_defaults)

        for col in ("Department", "Gender", "Education_Level"):
            if col in df.columns and df[col].notna().any():
                defaults[col] = str(df[col].mode().iloc[0]).strip()
    except Exception:
        return defaults

    return defaults


def load_dataset_scale_info():
    """Read dataset scale metadata used to normalize UI inputs and outputs."""
    info = {
        "performance_min": 0.0,
        "performance_max": 100.0,
        "satisfaction_min": 0.0,
        "satisfaction_max": 100.0,
    }
    data_path = os.path.join(BASE_DIR, "dataset", "employee_data.csv")
    if not os.path.exists(data_path):
        return info

    try:
        df = pd.read_csv(data_path)
        if "Performance_Score" in df.columns and df["Performance_Score"].notna().any():
            info["performance_min"] = float(df["Performance_Score"].min())
            info["performance_max"] = float(df["Performance_Score"].max())
        if "Employee_Satisfaction_Score" in df.columns and df["Employee_Satisfaction_Score"].notna().any():
            info["satisfaction_min"] = float(df["Employee_Satisfaction_Score"].min())
            info["satisfaction_max"] = float(df["Employee_Satisfaction_Score"].max())
    except Exception:
        return info

    return info


def load_metrics():
    """Load latest model metrics (regression + classification) from disk."""

    metrics_path = os.path.join(BASE_DIR, "model", "metrics.json")
    if not os.path.exists(metrics_path):
        return {}

    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def ensure_model_assets(force_retrain: bool = False):
    """Load model artifacts; if missing or forced, train using the dataset."""

    model_path = os.path.join(BASE_DIR, "productivity_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    features_path = os.path.join(BASE_DIR, "model_features.pkl")
    classifier_path = os.path.join(BASE_DIR, "burnout_classifier.pkl")
    label_encoder_path = classifier_path + ".labels"

    def is_lfs_pointer(path: str) -> bool:
        try:
            with open(path, "rb") as f:
                head = f.read(200)
            return b"git-lfs.github.com/spec" in head
        except Exception:
            return False

    need_training = force_retrain or not (
        os.path.exists(model_path)
        and os.path.exists(scaler_path)
        and os.path.exists(features_path)
        and os.path.exists(classifier_path)
        and os.path.exists(label_encoder_path)
    )

    # Retrain if metrics are from older training format/artifacts.
    if not need_training:
        metrics = load_metrics()
        reg = metrics.get("regression", {}) if isinstance(metrics, dict) else {}
        cls = metrics.get("classification", {}) if isinstance(metrics, dict) else {}
        metadata = metrics.get("metadata", {}) if isinstance(metrics, dict) else {}
        version = metadata.get("model_version")
        if (not reg.get("best_model")) or (not cls.get("best_model")) or (version != MODEL_VERSION):
            need_training = True
        if any(
            is_lfs_pointer(p)
            for p in (model_path, scaler_path, features_path, classifier_path, label_encoder_path)
        ):
            need_training = True

    if need_training:
        print("Training model artifacts using dataset...")
        train_and_save_model(
            data_path=os.path.join(BASE_DIR, "dataset", "employee_data.csv"),
            model_path=model_path,
            scaler_path=scaler_path,
            features_path=features_path,
            classifier_path=classifier_path,
            metrics_path=os.path.join(BASE_DIR, "model", "metrics.json"),
        )

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        classifier = joblib.load(classifier_path)
        label_encoder = joblib.load(label_encoder_path)
    except Exception as exc:
        print("Model load failed; retraining. Error:", exc)
        train_and_save_model(
            data_path=os.path.join(BASE_DIR, "dataset", "employee_data.csv"),
            model_path=model_path,
            scaler_path=scaler_path,
            features_path=features_path,
            classifier_path=classifier_path,
            metrics_path=os.path.join(BASE_DIR, "model", "metrics.json"),
        )
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        classifier = joblib.load(classifier_path)
        label_encoder = joblib.load(label_encoder_path)

    return model, scaler, features, classifier, label_encoder


model, scaler, features, classifier, label_encoder = ensure_model_assets()
BASELINE_FEATURE_VALUES = load_baseline_feature_values()
DATASET_SCALE_INFO = load_dataset_scale_info()


def reload_model_assets(retrain: bool = False):
    """Reload model artifacts into global variables."""
    global model, scaler, features, classifier, label_encoder
    model, scaler, features, classifier, label_encoder = ensure_model_assets(force_retrain=retrain)


# -------------------------------
# Activity Logging Helpers
# -------------------------------

def _get_request_ip():
    # Prefer reverse-proxy header when deployed; fallback to remote addr.
    return request.headers.get("X-Forwarded-For", request.remote_addr)


@app.before_request
def _log_employee_page_views():
    # Log only authenticated employee page activity (not static assets).
    if request.path.startswith("/static"):
        return
    if session.get("employee_id"):
        try:
            log_activity(
                actor_role="employee",
                employee_id=session.get("employee_id"),
                action="page_view",
                path=request.path,
                method=request.method,
                metadata=request.query_string.decode("utf-8") if request.query_string else None,
                ip_address=_get_request_ip(),
                user_agent=request.headers.get("User-Agent"),
            )
        except Exception:
            pass


# -------------------------------
# Access Control Decorators
# -------------------------------

def admin_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if not session.get("admin"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return wrap


def employee_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        # Allow both employees and admins to access prediction pages
        if not session.get("employee_id") and not session.get("admin"):
            return redirect(url_for("employee_login"))
        return f(*args, **kwargs)
    return wrap


# -------------------------------
# HOME
# -------------------------------

@app.route("/")
def index():
    # Redirect logged-in users to their home pages.
    if session.get("admin"):
        return redirect(url_for("admin_dashboard"))
    if session.get("employee_id"):
        return redirect(url_for("employee_dashboard"))
    return render_template("index.html")


# -------------------------------
# ADMIN LOGIN
# -------------------------------

@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    # Employees should not access admin login.
    if session.get("employee_id") and not session.get("admin"):
        return redirect(url_for("employee_dashboard"))

    if request.method == "POST":

        user = request.form.get("username")
        pw = request.form.get("password")

        if user == ADMIN_USER and pw == ADMIN_PASSWORD:

            session.clear()
            session["admin"] = True

            return redirect(url_for("admin_dashboard"))

    return render_template("admin_login.html")


# -------------------------------
# ADMIN DASHBOARD
# -------------------------------

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    # Pull data for dashboard stats and charts
    df = get_history(limit=100)
    stats = get_dashboard_stats()
    headcount = get_unique_employee_count()

    # Build Plotly charts
    dept_avgs = get_department_averages()
    risk_counts = get_risk_counts()
    recent_preds = list(reversed(get_recent_predictions(20)))  # oldest-first for line chart

    import plotly.graph_objects as go

    dept_names = [r["department"] or "Unknown" for r in dept_avgs]
    dept_scores = [round(r["avg_score"], 2) if r["avg_score"] is not None else 0 for r in dept_avgs]

    fig_dept = go.Figure(
        go.Bar(x=dept_names, y=dept_scores, marker_color="#4f46e5")
    )
    fig_dept.update_layout(
        title="Avg. Productivity by Department",
        xaxis_title="Department",
        yaxis_title="Avg. Productivity",
        template="plotly_white",
    )

    fig_risk = go.Figure(
        go.Pie(
            labels=["High", "Moderate", "Low"],
            values=[risk_counts["high"], risk_counts["moderate"], risk_counts["low"]],
            hole=0.4,
        )
    )
    fig_risk.update_layout(title="Burnout Risk Distribution", template="plotly_white")

    fig_trend = go.Figure(
        go.Scatter(
            x=[r["id"] for r in recent_preds],
            y=[r["result"] for r in recent_preds],
            mode="lines+markers",
            marker=dict(color="#0ea5e9"),
            line=dict(shape="spline", smoothing=0.6),
        )
    )
    fig_trend.update_layout(
        title="Recent Productivity Scores",
        xaxis_title="Record ID",
        yaxis_title="Predicted Productivity",
        template="plotly_white",
    )

    charts = [fig_to_json_obj(fig) for fig in (fig_dept, fig_risk, fig_trend)]

    # If there is no prediction data yet, show a placeholder chart rather than blank cards.
    if stats.get("total_predictions", 0) == 0:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No prediction data yet — ask employees to run an assessment.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            font=dict(size=18, color="#64748b"),
        )
        empty_fig.update_layout(template="plotly_white")
        charts = [fig_to_json_obj(empty_fig)]
        chart_drilldowns = [None]

    return render_template(
        "dashboard_admin.html",
        df=df,
        charts=charts,
        stats=stats,
        headcount=headcount,
    )


# -------------------------------
# ADMIN MODEL METRICS
@app.route("/admin/model-metrics")
@admin_required
def admin_model_metrics():
    """Show model training metrics (regression + classification)."""

    metrics = load_metrics()
    retrained = request.args.get("retrained") == "1"

    return render_template(
        "admin/model_metrics.html",
        metrics=metrics,
        retrained=retrained,
    )


@app.route("/admin/retrain", methods=["POST"])
@admin_required
def admin_retrain():
    """Retrain models on demand and reload artifacts."""

    reload_model_assets(retrain=True)

    return redirect(url_for("admin_model_metrics", retrained=1))


# -------------------------------
# ADMIN ANALYTICS PAGE
# -------------------------------

@app.route("/admin/analytics")
@admin_required
def admin_analytics():
    """Render an expanded analytics dashboard with multiple charts."""

    # Filter configuration (time range and optional drilldowns)
    days = request.args.get("days")
    employee_id = normalize_filter(request.args.get("employee_id"))
    department = normalize_filter(request.args.get("department"))
    risk = normalize_risk_filter(request.args.get("risk"))
    team_size_band = normalize_band(request.args.get("team_size_band"), ["Small", "Medium", "Large"])
    experience_band = normalize_band(request.args.get("experience_band"), ["Junior", "Mid", "Senior", "Veteran"])
    salary_band = normalize_band(request.args.get("salary_band"), ["Low", "Mid", "High"])
    remote_band = normalize_band(request.args.get("remote_band"), ["Onsite", "Hybrid", "Remote"])
    try:
        days = int(days) if days not in (None, "") else None
    except ValueError:
        days = None

    # Use recent prediction history for chart generation
    history_rows = get_history(
        limit=5000,
        days=days,
        emp_id=employee_id,
        department=department,
        risk=risk,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
    )

    # Convert sqlite Row objects into dicts for easier handling
    import pandas as pd

    df = pd.DataFrame([dict(r) for r in history_rows]) if history_rows else pd.DataFrame()

    # Build charts using Plotly
    import plotly.graph_objects as go

    charts = []
    chart_drilldowns = []

    # Cohort comparison (current vs previous window)
    compare_days = days or 30
    cohort_chart = None
    insights = []
    anomalies = []
    dept_goal_rows = []

    goals = get_department_goals()
    if not df.empty and "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        now = pd.Timestamp.now()
        current_start = now - pd.Timedelta(days=compare_days)
        previous_start = now - pd.Timedelta(days=compare_days * 2)

        current_df = df[df["created_at"] >= current_start]
        previous_df = df[(df["created_at"] < current_start) & (df["created_at"] >= previous_start)]

        current_dept = (
            current_df.groupby("department")["result"].mean().dropna()
            if not current_df.empty and "department" in current_df.columns and "result" in current_df.columns
            else pd.Series(dtype=float)
        )
        previous_dept = (
            previous_df.groupby("department")["result"].mean().dropna()
            if not previous_df.empty and "department" in previous_df.columns and "result" in previous_df.columns
            else pd.Series(dtype=float)
        )

        all_departments = set(current_dept.index.astype(str)).union(set(previous_dept.index.astype(str)))
        all_departments.update([d for d in goals.keys() if d])
        all_departments = sorted(all_departments)
        if all_departments:
            current_vals = [round(float(current_dept.get(d, 0)), 2) for d in all_departments]
            previous_vals = [round(float(previous_dept.get(d, 0)), 2) for d in all_departments]
            fig_cohort = go.Figure()
            fig_cohort.add_bar(name=f"Last {compare_days} days", x=all_departments, y=current_vals)
            fig_cohort.add_bar(name=f"Previous {compare_days} days", x=all_departments, y=previous_vals)
            fig_cohort.update_layout(
                title=f"Department Productivity: Current vs Previous {compare_days} Days",
                barmode="group",
                xaxis_title="Department",
                yaxis_title="Avg Productivity",
                template="plotly_white",
            )
            cohort_chart = fig_to_json_obj(fig_cohort)

        # Insights
        if not current_df.empty and "result" in current_df.columns:
            current_avg = float(current_df["result"].mean())
            previous_avg = float(previous_df["result"].mean()) if not previous_df.empty else None
            if previous_avg is not None and previous_avg > 0:
                delta = current_avg - previous_avg
                insights.append(
                    f"Overall productivity is {current_avg:.2f}% vs {previous_avg:.2f}% in the previous {compare_days} days ({delta:+.2f} pts)."
                )

        if not current_dept.empty and not previous_dept.empty:
            deltas = {}
            for dept in all_departments:
                if dept in current_dept.index and dept in previous_dept.index:
                    deltas[dept] = float(current_dept.get(dept, 0)) - float(previous_dept.get(dept, 0))
            if deltas:
                top_up = max(deltas.items(), key=lambda kv: kv[1])
                top_down = min(deltas.items(), key=lambda kv: kv[1])
                if top_up[1] > 0:
                    insights.append(
                        f"{top_up[0]} productivity up {top_up[1]:.2f} pts vs previous {compare_days} days."
                    )
                if top_down[1] < 0:
                    insights.append(
                        f"{top_down[0]} productivity down {abs(top_down[1]):.2f} pts vs previous {compare_days} days."
                    )

        if not current_df.empty and "risk" in current_df.columns:
            current_high = (current_df["risk"].fillna("").str.startswith("High").mean()) * 100
            previous_high = (previous_df["risk"].fillna("").str.startswith("High").mean()) * 100 if not previous_df.empty else None
            if previous_high is not None:
                delta = current_high - previous_high
                insights.append(
                    f"High-risk share is {current_high:.1f}% vs {previous_high:.1f}% ({delta:+.1f} pts)."
                )

        # Anomaly detection by department
        if all_departments:
            for dept in all_departments:
                cur_vals = current_df[current_df["department"] == dept]["result"] if "department" in current_df.columns else pd.Series(dtype=float)
                prev_vals = previous_df[previous_df["department"] == dept]["result"] if "department" in previous_df.columns else pd.Series(dtype=float)
                if len(cur_vals) >= 3 and len(prev_vals) >= 3:
                    cur_avg = float(cur_vals.mean())
                    prev_avg = float(prev_vals.mean())
                    delta = cur_avg - prev_avg
                    if delta <= -8:
                        anomalies.append(
                            {"department": dept, "type": "productivity_drop", "message": f"{dept}: productivity down {abs(delta):.1f} pts vs previous period."}
                        )

                cur_high = (current_df[current_df["department"] == dept]["risk"].fillna("").str.startswith("High").mean()) if "risk" in current_df.columns else None
                prev_high = (previous_df[previous_df["department"] == dept]["risk"].fillna("").str.startswith("High").mean()) if "risk" in previous_df.columns else None
                if cur_high is not None and prev_high is not None:
                    delta_high = (cur_high - prev_high) * 100
                    if delta_high >= 10:
                        anomalies.append(
                            {"department": dept, "type": "risk_spike", "message": f"{dept}: high-risk share up {delta_high:.1f} pts."}
                        )

        # Goal tracking rows
        for dept in all_departments:
            target = goals.get(dept)
            current_avg = float(current_dept.get(dept, 0)) if dept in current_dept.index else None
            progress = None
            if target and current_avg is not None and target > 0:
                progress = min(150, (current_avg / target) * 100)
            dept_goal_rows.append(
                {
                    "department": dept,
                    "target": target,
                    "current": current_avg,
                    "progress": progress,
                }
            )
    else:
        # Build goal rows from saved goals even if there's no data yet.
        for dept, target in goals.items():
            dept_goal_rows.append(
                {
                    "department": dept,
                    "target": target,
                    "current": None,
                    "progress": None,
                }
            )

    # 1) Average productivity by department
    dept_avgs = get_department_averages(
        days=days,
        emp_id=employee_id,
        department=department,
        risk=risk,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
    )
    dept_names = [r["department"] or "Unknown" for r in dept_avgs]
    dept_scores = [round(r["avg_score"], 2) if r["avg_score"] is not None else 0 for r in dept_avgs]

    fig_dept = go.Figure(go.Bar(x=dept_names, y=dept_scores, marker_color="#4f46e5"))
    fig_dept.update_layout(
        title="Avg. Productivity by Department",
        xaxis_title="Department",
        yaxis_title="Avg. Productivity",
        template="plotly_white",
    )
    charts.append(fig_to_json_obj(fig_dept))
    chart_drilldowns.append("department")

    # 2) Burnout risk distribution
    risk_counts = get_risk_counts(
        days=days,
        emp_id=employee_id,
        department=department,
        risk=risk,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
    )
    fig_risk = go.Figure(
        go.Pie(
            labels=["High", "Moderate", "Low"],
            values=[risk_counts["high"], risk_counts["moderate"], risk_counts["low"]],
            hole=0.4,
        )
    )
    fig_risk.update_layout(title="Burnout Risk Distribution", template="plotly_white")
    charts.append(fig_to_json_obj(fig_risk))
    chart_drilldowns.append("risk")

    # 3) Recent productivity trend (by record id)
    if not df.empty and "id" in df.columns and "result" in df.columns:
        fig_trend = go.Figure(
            go.Scatter(
                x=df["id"],
                y=df["result"],
                mode="lines+markers",
                marker=dict(color="#0ea5e9"),
                line=dict(shape="spline", smoothing=0.6),
            )
        )
        fig_trend.update_layout(
            title="Recent Productivity Scores (Trend)",
            xaxis_title="Record ID",
            yaxis_title="Predicted Productivity",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_trend))
        chart_drilldowns.append(None)

    # 4) Productivity distribution histogram
    if not df.empty and "result" in df.columns:
        fig_hist = go.Figure(
            go.Histogram(x=df["result"], nbinsx=20, marker_color="#2563eb")
        )
        fig_hist.update_layout(
            title="Productivity Score Distribution",
            xaxis_title="Predicted Productivity",
            yaxis_title="Count",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_hist))
        chart_drilldowns.append(None)

    # 5) Salary vs Productivity scatter
    if not df.empty and "monthly_salary" in df.columns and "result" in df.columns:
        fig_salary = go.Figure(
            go.Scatter(
                x=df["monthly_salary"],
                y=df["result"],
                mode="markers",
                marker=dict(color="#059669", size=8, opacity=0.7),
            )
        )
        fig_salary.update_layout(
            title="Salary vs Productivity",
            xaxis_title="Monthly Salary",
            yaxis_title="Predicted Productivity",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_salary))
        chart_drilldowns.append(None)

    # 6) Satisfaction vs Productivity scatter
    if not df.empty and "satisfaction_score" in df.columns and "result" in df.columns:
        fig_sat = go.Figure(
            go.Scatter(
                x=df["satisfaction_score"],
                y=df["result"],
                mode="markers",
                marker=dict(color="#f97316", size=8, opacity=0.7),
            )
        )
        fig_sat.update_layout(
            title="Satisfaction vs Productivity",
            xaxis_title="Satisfaction Score",
            yaxis_title="Predicted Productivity",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_sat))
        chart_drilldowns.append(None)

    # 7) Overtime / Work Hours vs Productivity scatter
    if not df.empty and "work_hours" in df.columns and "result" in df.columns:
        fig_overtime = go.Figure(
            go.Scatter(
                x=df["work_hours"],
                y=df["result"],
                mode="markers",
                marker=dict(color="#be123c", size=8, opacity=0.7),
            )
        )
        fig_overtime.update_layout(
            title="Work Hours vs Productivity",
            xaxis_title="Work Hours",
            yaxis_title="Predicted Productivity",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_overtime))
        chart_drilldowns.append(None)

    # 8) Top employees by average productivity
    if not df.empty and "employee_id" in df.columns and "result" in df.columns:
        top_emps = (
            df.groupby("employee_id")["result"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        fig_top = go.Figure(
            go.Bar(x=top_emps.index.astype(str), y=top_emps.values, marker_color="#0ea5e9")
        )
        fig_top.update_layout(
            title="Top 10 Employees by Avg. Productivity",
            xaxis_title="Employee ID",
            yaxis_title="Avg Productivity",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_top))
        chart_drilldowns.append("employee")

    # 9) Predictions count by department
    if not df.empty and "department" in df.columns:
        dept_counts = df["department"].fillna("Unknown").value_counts().head(10)
        fig_dept_count = go.Figure(
            go.Bar(x=dept_counts.index, y=dept_counts.values, marker_color="#2563eb")
        )
        fig_dept_count.update_layout(
            title="Predictions by Department",
            xaxis_title="Department",
            yaxis_title="Count",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_dept_count))
        chart_drilldowns.append("department")

    # 10) Risk category breakdown by count
    if not df.empty and "risk" in df.columns:
        risk_counts = df["risk"].fillna("Unknown").value_counts()
        fig_risk_count = go.Figure(
            go.Bar(x=risk_counts.index, y=risk_counts.values, marker_color="#f97316")
        )
        fig_risk_count.update_layout(
            title="Prediction Risk Category Counts",
            xaxis_title="Risk Category",
            yaxis_title="Count",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_risk_count))
        chart_drilldowns.append("risk")

    # 11) Productivity by work hours bucket (box plot)
    if not df.empty and "work_hours" in df.columns and "result" in df.columns:
        # Handle low-variance filtered slices (single employee) without raising bin-edge errors.
        if df["work_hours"].nunique() > 1:
            df["hours_bucket"] = pd.cut(
                df["work_hours"],
                bins=5,
                labels=["Very Low", "Low", "Medium", "High", "Very High"],
                duplicates="drop",
            )
        else:
            df["hours_bucket"] = "Single Range"
        fig_box = go.Figure(
            go.Box(
                x=df["hours_bucket"],
                y=df["result"],
                marker_color="#14b8a6",
            )
        )
        fig_box.update_layout(
            title="Productivity by Work Hours Bucket",
            xaxis_title="Work Hours Bucket",
            yaxis_title="Predicted Productivity",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_box))
        chart_drilldowns.append(None)

    # 12) Salary distribution (box plot)
    if not df.empty and "monthly_salary" in df.columns:
        fig_salary_box = go.Figure(
            go.Box(y=df["monthly_salary"], marker_color="#a855f7")
        )
        fig_salary_box.update_layout(
            title="Salary Distribution",
            yaxis_title="Monthly Salary",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_salary_box))
        chart_drilldowns.append(None)

    # If there is no data in the selected window, show a helpful placeholder chart.
    if not charts:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No predictions found for the selected filters.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            font=dict(size=18, color="#64748b"),
        )
        empty_fig.update_layout(template="plotly_white")
        charts = [fig_to_json_obj(empty_fig)]

    departments = []
    if not df.empty and "department" in df.columns:
        departments = df["department"].dropna().astype(str).unique().tolist()
        departments = sorted([d for d in departments if d.strip()])

    # Compare employees: Top 5 vs Bottom 5
    top_bottom = {"top": [], "bottom": [], "deltas": []}
    feature_map = [
        ("monthly_salary", "Monthly Salary (k)"),
        ("work_hours", "Work Hours"),
        ("satisfaction_score", "Satisfaction Score"),
        ("projects_handled", "Projects Handled"),
        ("team_size", "Team Size"),
        ("years_at_company", "Years at Company"),
        ("remote_work_frequency", "Remote Frequency"),
    ]

    if not df.empty and "employee_id" in df.columns and "result" in df.columns:
        grouped = df.groupby("employee_id")
        summary = grouped["result"].mean().sort_values(ascending=False)
        top_ids = summary.head(5).index.tolist()
        bottom_ids = summary.tail(5).index.tolist()

        def build_group_rows(ids):
            rows = []
            for emp_id in ids:
                emp_rows = df[df["employee_id"] == emp_id]
                emp_name = emp_rows["employee_name"].dropna().astype(str).iloc[0] if "employee_name" in emp_rows.columns and not emp_rows["employee_name"].dropna().empty else ""
                dept = emp_rows["department"].dropna().astype(str).iloc[0] if "department" in emp_rows.columns and not emp_rows["department"].dropna().empty else ""
                row = {
                    "employee_id": emp_id,
                    "employee_name": emp_name,
                    "department": dept,
                    "avg_score": round(float(summary.get(emp_id, 0)), 2),
                }
                for key, _label in feature_map:
                    if key in emp_rows.columns and emp_rows[key].notna().any():
                        row[key] = round(float(emp_rows[key].mean()), 2)
                    else:
                        row[key] = None
                rows.append(row)
            return rows

        top_rows = build_group_rows(top_ids)
        bottom_rows = build_group_rows(bottom_ids)
        top_bottom["top"] = top_rows
        top_bottom["bottom"] = bottom_rows

        if top_rows and bottom_rows:
            top_df = pd.DataFrame(top_rows)
            bottom_df = pd.DataFrame(bottom_rows)
            for key, label in feature_map:
                if key in top_df.columns and key in bottom_df.columns:
                    top_avg = top_df[key].dropna().mean()
                    bottom_avg = bottom_df[key].dropna().mean()
                    if pd.notna(top_avg) and pd.notna(bottom_avg):
                        top_bottom["deltas"].append(
                            {
                                "label": label,
                                "top_avg": round(float(top_avg), 2),
                                "bottom_avg": round(float(bottom_avg), 2),
                                "delta": round(float(top_avg - bottom_avg), 2),
                            }
                        )

    # Confidence visuals (model + data quality)
    metrics = load_metrics()
    regression_metrics = metrics.get("regression", {})
    model_confidence = None
    if regression_metrics:
        accuracy_pct = regression_metrics.get("accuracy_pct")
        if accuracy_pct is not None:
            model_confidence = float(accuracy_pct)
        else:
            reg_r2 = float(regression_metrics.get("r2_score", 0) or 0)
            model_confidence = round(max(0.0, min(100.0, reg_r2 * 100)), 2)

    data_quality = None
    if not df.empty:
        quality_fields = [k for k, _ in feature_map if k in df.columns]
        if quality_fields:
            total = len(df) * len(quality_fields)
            non_null = int(df[quality_fields].notna().sum().sum())
            if total > 0:
                data_quality = round((non_null / total) * 100, 2)

    analytics_query = build_analytics_query(
        days=days,
        employee_id=employee_id,
        department=department,
        risk=risk,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
    )
    analytics_url = url_for("admin_analytics", **analytics_query)

    return render_template(
        "admin/analytics_admin.html",
        charts=charts,
        df=df,
        days=days,
        employee_id=employee_id,
        department=department,
        risk=risk,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
        departments=departments,
        risk_options=["High", "Moderate", "Low"],
        band_options={
            "team_size": ["Small", "Medium", "Large"],
            "experience": ["Junior", "Mid", "Senior", "Veteran"],
            "salary": ["Low", "Mid", "High"],
            "remote": ["Onsite", "Hybrid", "Remote"],
        },
        cohort_chart=cohort_chart,
        compare_days=compare_days,
        insights=insights,
        anomalies=anomalies,
        goal_rows=dept_goal_rows,
        analytics_url=analytics_url,
        chart_drilldowns=chart_drilldowns,
        top_bottom=top_bottom,
        model_confidence=model_confidence,
        data_quality=data_quality,
    )


@app.route("/admin/analytics/goals", methods=["POST"])
@admin_required
def admin_analytics_goals():
    """Persist department productivity targets."""

    for key, value in request.form.items():
        if not key.startswith("goal_"):
            continue
        dept = key.replace("goal_", "").strip()
        if not dept:
            continue
        val = str(value).strip()
        if val == "":
            continue
        try:
            target = float(val)
        except ValueError:
            continue
        upsert_department_goal(dept, target)

    next_url = request.form.get("next") or url_for("admin_analytics")
    return redirect(next_url)


@app.route("/admin/analytics/chart/<int:chart_index>")
@admin_required
def admin_analytics_chart(chart_index: int):
    """Dedicated fullscreen chart view with a shareable URL."""

    days = request.args.get("days")
    employee_id = normalize_filter(request.args.get("employee_id"))
    department = normalize_filter(request.args.get("department"))
    risk = normalize_risk_filter(request.args.get("risk"))
    team_size_band = normalize_band(request.args.get("team_size_band"), ["Small", "Medium", "Large"])
    experience_band = normalize_band(request.args.get("experience_band"), ["Junior", "Mid", "Senior", "Veteran"])
    salary_band = normalize_band(request.args.get("salary_band"), ["Low", "Mid", "High"])
    remote_band = normalize_band(request.args.get("remote_band"), ["Onsite", "Hybrid", "Remote"])
    try:
        days = int(days) if days not in (None, "") else None
    except ValueError:
        days = None

    history_rows = get_history(
        limit=5000,
        days=days,
        emp_id=employee_id,
        department=department,
        risk=risk,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
    )

    import pandas as pd
    import plotly.graph_objects as go

    df = pd.DataFrame([dict(r) for r in history_rows]) if history_rows else pd.DataFrame()
    charts = []

    dept_avgs = get_department_averages(
        days=days,
        emp_id=employee_id,
        department=department,
        risk=risk,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
    )
    dept_names = [r["department"] or "Unknown" for r in dept_avgs]
    dept_scores = [round(r["avg_score"], 2) if r["avg_score"] is not None else 0 for r in dept_avgs]
    fig_dept = go.Figure(go.Bar(x=dept_names, y=dept_scores, marker_color="#4f46e5"))
    fig_dept.update_layout(
        title="Avg. Productivity by Department",
        xaxis_title="Department",
        yaxis_title="Avg. Productivity",
        template="plotly_white",
    )
    charts.append(fig_to_json_obj(fig_dept))
    chart_drilldowns.append("department")

    risk_counts = get_risk_counts(
        days=days,
        emp_id=employee_id,
        department=department,
        risk=risk,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
    )
    fig_risk = go.Figure(
        go.Pie(
            labels=["High", "Moderate", "Low"],
            values=[risk_counts["high"], risk_counts["moderate"], risk_counts["low"]],
            hole=0.4,
        )
    )
    fig_risk.update_layout(title="Burnout Risk Distribution", template="plotly_white")
    charts.append(fig_to_json_obj(fig_risk))
    chart_drilldowns.append("risk")

    if not df.empty and "id" in df.columns and "result" in df.columns:
        fig_trend = go.Figure(
            go.Scatter(
                x=df["id"],
                y=df["result"],
                mode="lines+markers",
                marker=dict(color="#0ea5e9"),
                line=dict(shape="spline", smoothing=0.6),
            )
        )
        fig_trend.update_layout(
            title="Recent Productivity Scores (Trend)",
            xaxis_title="Record ID",
            yaxis_title="Predicted Productivity",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_trend))
        chart_drilldowns.append(None)

    if not df.empty and "result" in df.columns:
        fig_hist = go.Figure(
            go.Histogram(x=df["result"], nbinsx=20, marker_color="#2563eb")
        )
        fig_hist.update_layout(
            title="Productivity Score Distribution",
            xaxis_title="Predicted Productivity",
            yaxis_title="Count",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_hist))
        chart_drilldowns.append(None)

    if not df.empty and "monthly_salary" in df.columns and "result" in df.columns:
        fig_salary = go.Figure(
            go.Scatter(
                x=df["monthly_salary"],
                y=df["result"],
                mode="markers",
                marker=dict(color="#059669", size=8, opacity=0.7),
            )
        )
        fig_salary.update_layout(
            title="Salary vs Productivity",
            xaxis_title="Monthly Salary",
            yaxis_title="Predicted Productivity",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_salary))
        chart_drilldowns.append(None)

    if not df.empty and "satisfaction_score" in df.columns and "result" in df.columns:
        fig_sat = go.Figure(
            go.Scatter(
                x=df["satisfaction_score"],
                y=df["result"],
                mode="markers",
                marker=dict(color="#f97316", size=8, opacity=0.7),
            )
        )
        fig_sat.update_layout(
            title="Satisfaction vs Productivity",
            xaxis_title="Satisfaction Score",
            yaxis_title="Predicted Productivity",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_sat))
        chart_drilldowns.append(None)

    if not df.empty and "work_hours" in df.columns and "result" in df.columns:
        fig_overtime = go.Figure(
            go.Scatter(
                x=df["work_hours"],
                y=df["result"],
                mode="markers",
                marker=dict(color="#be123c", size=8, opacity=0.7),
            )
        )
        fig_overtime.update_layout(
            title="Work Hours vs Productivity",
            xaxis_title="Work Hours",
            yaxis_title="Predicted Productivity",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_overtime))
        chart_drilldowns.append(None)

    if not df.empty and "employee_id" in df.columns and "result" in df.columns:
        top_emps = (
            df.groupby("employee_id")["result"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        fig_top = go.Figure(
            go.Bar(x=top_emps.index.astype(str), y=top_emps.values, marker_color="#0ea5e9")
        )
        fig_top.update_layout(
            title="Top 10 Employees by Avg. Productivity",
            xaxis_title="Employee ID",
            yaxis_title="Avg Productivity",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_top))
        chart_drilldowns.append("employee")

    if not df.empty and "department" in df.columns:
        dept_counts = df["department"].fillna("Unknown").value_counts().head(10)
        fig_dept_count = go.Figure(
            go.Bar(x=dept_counts.index, y=dept_counts.values, marker_color="#2563eb")
        )
        fig_dept_count.update_layout(
            title="Predictions by Department",
            xaxis_title="Department",
            yaxis_title="Count",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_dept_count))
        chart_drilldowns.append("department")

    if not df.empty and "risk" in df.columns:
        risk_series = df["risk"].fillna("Unknown").value_counts()
        fig_risk_count = go.Figure(
            go.Bar(x=risk_series.index, y=risk_series.values, marker_color="#f97316")
        )
        fig_risk_count.update_layout(
            title="Prediction Risk Category Counts",
            xaxis_title="Risk Category",
            yaxis_title="Count",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_risk_count))
        chart_drilldowns.append("risk")

    if not df.empty and "work_hours" in df.columns and "result" in df.columns:
        if df["work_hours"].nunique() > 1:
            df["hours_bucket"] = pd.cut(
                df["work_hours"],
                bins=5,
                labels=["Very Low", "Low", "Medium", "High", "Very High"],
                duplicates="drop",
            )
        else:
            df["hours_bucket"] = "Single Range"
        fig_box = go.Figure(
            go.Box(
                x=df["hours_bucket"],
                y=df["result"],
                marker_color="#14b8a6",
            )
        )
        fig_box.update_layout(
            title="Productivity by Work Hours Bucket",
            xaxis_title="Work Hours Bucket",
            yaxis_title="Predicted Productivity",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_box))
        chart_drilldowns.append(None)

    if not df.empty and "monthly_salary" in df.columns:
        fig_salary_box = go.Figure(
            go.Box(y=df["monthly_salary"], marker_color="#a855f7")
        )
        fig_salary_box.update_layout(
            title="Salary Distribution",
            yaxis_title="Monthly Salary",
            template="plotly_white",
        )
        charts.append(fig_to_json_obj(fig_salary_box))
        chart_drilldowns.append(None)

    if not charts:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No predictions found in the selected filters.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            font=dict(size=18, color="#64748b"),
        )
        empty_fig.update_layout(template="plotly_white")
        charts.append(fig_to_json_obj(empty_fig))

    if chart_index < 1 or chart_index > len(charts):
        return redirect(url_for("admin_analytics"))

    selected_chart = charts[chart_index - 1]
    chart_title = chart_title_from_json(selected_chart, f"Chart {chart_index}")

    return render_template(
        "admin/analytics_chart.html",
        chart_json=selected_chart,
        chart_title=chart_title,
        chart_index=chart_index,
        days=days,
        employee_id=employee_id,
        department=department,
        risk=risk,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
    )


@app.route("/admin/employees")
@admin_required
def admin_employees():
    """Focused employee list with quick actions."""

    days = request.args.get("days")
    employee_id = normalize_filter(request.args.get("employee_id"))
    department = normalize_filter(request.args.get("department"))
    risk = normalize_risk_filter(request.args.get("risk"))
    team_size_band = normalize_band(request.args.get("team_size_band"), ["Small", "Medium", "Large"])
    experience_band = normalize_band(request.args.get("experience_band"), ["Junior", "Mid", "Senior", "Veteran"])
    salary_band = normalize_band(request.args.get("salary_band"), ["Low", "Mid", "High"])
    remote_band = normalize_band(request.args.get("remote_band"), ["Onsite", "Hybrid", "Remote"])
    try:
        days = int(days) if days not in (None, "") else None
    except ValueError:
        days = None

    rows = get_employee_summary(
        days=days,
        department=department,
        risk=risk,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
    )

    if employee_id:
        rows = [r for r in rows if str(r["employee_id"]) == str(employee_id)]

    analytics_query = build_analytics_query(
        days=days,
        employee_id=employee_id,
        department=department,
        risk=risk,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
    )

    return render_template(
        "admin/employee_list.html",
        employees=rows,
        days=days,
        employee_id=employee_id,
        department=department,
        risk=risk,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
        analytics_query=analytics_query,
    )


@app.route("/admin/activity")
@admin_required
def admin_activity():
    """Audit log of employee activity."""

    days = request.args.get("days")
    employee_id = normalize_filter(request.args.get("employee_id"))
    action = normalize_filter(request.args.get("action"))
    try:
        days = int(days) if days not in (None, "") else None
    except ValueError:
        days = None

    rows = get_activity_log(employee_id=employee_id, days=days, action=action, limit=500)

    return render_template(
        "admin/activity_log.html",
        activity_rows=rows,
        days=days,
        employee_id=employee_id,
        action=action,
        action_options=["login", "logout", "page_view", "prediction_submitted"],
    )


# -------------------------------
# ADMIN ALERTS PAGE
# -------------------------------

@app.route("/admin/alerts")
@admin_required
def admin_alerts():

    alerts = burnout_alerts()

    return render_template("admin/alert_admin.html", alerts=alerts)


# -------------------------------
# EMPLOYEE LOGIN
# -------------------------------

@app.route("/employee_login", methods=["GET", "POST"])
def employee_login():

    if request.method == "POST":

        emp_id = request.form.get("employee_id")

        session.clear()
        session["employee_id"] = emp_id

        try:
            log_activity(
                actor_role="employee",
                employee_id=emp_id,
                action="login",
                path=request.path,
                method=request.method,
                ip_address=_get_request_ip(),
                user_agent=request.headers.get("User-Agent"),
            )
        except Exception:
            pass

        return redirect(url_for("employee_dashboard"))

    return render_template("employee_login.html")


# -------------------------------
# EMPLOYEE DASHBOARD
# -------------------------------

@app.route("/employee/dashboard")
@employee_required
def employee_dashboard():

    return render_template("dashboard_employee.html")


# -------------------------------
# ADVANCED PREDICTION MODE
# -------------------------------

@app.route("/predict_advanced")
@employee_required
def predict_advanced():
    departments = [
        "Engineering",
        "Finance",
        "HR",
        "IT",
        "Legal",
        "Marketing",
        "Operations",
        "Sales",
    ]

    genders = ["Male", "Female", "Other"]
    education_levels = ["High School", "Bachelor", "Master", "PhD"]

    return render_template(
        "predict_advanced.html",
        departments=departments,
        genders=genders,
        education_levels=education_levels,
    )


# -------------------------------
# EMPLOYEE PREDICTION
# -------------------------------

def build_feature_vector(form):
    """Build a feature vector dict matching the trained model features."""

    # Start with zeros for all known model features
    data = {f: 0.0 for f in features}
    # Seed with realistic baseline values for fields not provided in quick mode.
    for key, value in BASELINE_FEATURE_VALUES.items():
        if key in data:
            try:
                data[key] = float(value)
            except (TypeError, ValueError):
                pass

    # Numeric inputs that map directly
    numeric_fields = [
        "Age",
        "Years_At_Company",
        "Monthly_Salary",
        "Work_Hours_Per_Week",
        "Projects_Handled",
        "Overtime_Hours",
        "Sick_Days",
        "Remote_Work_Frequency",
        "Team_Size",
        "Training_Hours",
        "Promotions",
        "Employee_Satisfaction_Score",
    ]

    for f in numeric_fields:
        if f in data:
            v = form.get(f)
            if v not in (None, ""):
                try:
                    data[f] = float(v)
                except ValueError:
                    data[f] = 0.0

    # Feature engineering: scale salary to thousands (20,000 -> 20) for user-entered values.
    if "Monthly_Salary" in data and form.get("Monthly_Salary") not in (None, ""):
        try:
            data["Monthly_Salary"] = float(data["Monthly_Salary"]) / 1000.0
        except Exception:
            pass

    # Handle binary/boolean fields
    if "Resigned" in data:
        val = form.get("Resigned")
        if val in ("1", "true", "True", "yes", "Yes"):
            data["Resigned"] = 1.0
        else:
            data["Resigned"] = 0.0

    # Back-compat for the quick form field name
    if "Employee_Satisfaction_Score" in data:
        alt = form.get("Satisfaction_Score")
        if alt not in (None, ""):
            try:
                data["Employee_Satisfaction_Score"] = float(alt)
            except ValueError:
                pass

    # Normalize satisfaction scale to training-data range (dataset is often ~1-5).
    if "Employee_Satisfaction_Score" in data:
        sat = float(data.get("Employee_Satisfaction_Score", 0) or 0)
        sat_max = float(DATASET_SCALE_INFO.get("satisfaction_max", 100.0) or 100.0)
        sat_min = float(DATASET_SCALE_INFO.get("satisfaction_min", 0.0) or 0.0)
        if sat_max <= 10 and sat > 10:
            sat = sat / 20.0  # map 0-100 UI input to 0-5-like training scale
        data["Employee_Satisfaction_Score"] = float(np.clip(sat, sat_min, sat_max))

    # Derived features
    total_effort = data.get("Work_Hours_Per_Week", 0) + data.get("Overtime_Hours", 0)
    data["Total_Effort"] = total_effort

    projects = data.get("Projects_Handled", 0)
    data["Effort_per_Project"] = total_effort / max(1.0, projects)

    team_size = data.get("Team_Size", 1)
    data["Workload_Index"] = total_effort / max(1.0, team_size)

    # Categoricals -> one-hot features
    department = (form.get("Department", "") or BASELINE_FEATURE_VALUES.get("Department", "")).strip()
    gender = (form.get("Gender", "") or BASELINE_FEATURE_VALUES.get("Gender", "")).strip()
    education = (form.get("Education_Level", "") or BASELINE_FEATURE_VALUES.get("Education_Level", "")).strip()

    for feat in data:
        if feat.startswith("Department_"):
            data[feat] = 1.0 if feat == f"Department_{department}" else 0.0
        if feat.startswith("Gender_"):
            data[feat] = 1.0 if feat == f"Gender_{gender}" else 0.0
        if feat.startswith("Education_Level_"):
            data[feat] = 1.0 if feat == f"Education_Level_{education}" else 0.0

    return data


@app.route("/predict", methods=["GET", "POST"])
@employee_required
def employee_predict():

    if request.method == "POST":

        try:
            # Build complete feature set using available inputs
            feature_dict = build_feature_vector(request.form)

            df = pd.DataFrame([feature_dict])
            df = df.reindex(columns=features, fill_value=0)

            scaled = scaler.transform(df)
            scaled_df = pd.DataFrame(scaled, columns=features)

            prediction_raw = float(model.predict(scaled)[0])
            perf_min = float(DATASET_SCALE_INFO.get("performance_min", 0.0) or 0.0)
            perf_max = float(DATASET_SCALE_INFO.get("performance_max", 100.0) or 100.0)
            if perf_max <= 10:
                # Convert model-native scale (e.g., 1..5) into UI percentage.
                final_score = round(float(np.clip((prediction_raw / perf_max) * 100.0, 0, 100)), 2)
            else:
                final_score = round(float(np.clip(prediction_raw, 0, 100)), 2)

            # Classifier prediction (burnout risk)
            burnout_confidence = None
            try:
                risk_idx = classifier.predict(scaled)[0]
                risk = label_encoder.inverse_transform([risk_idx])[0]
                if hasattr(classifier, "predict_proba"):
                    proba = classifier.predict_proba(scaled)[0]
                    burnout_confidence = round(float(np.max(proba) * 100), 2)
            except Exception:
                # Fall back to rule-based risk if classifier fails
                risk = burnout_risk(
                    float(feature_dict.get("Work_Hours_Per_Week", 0)),
                    float(feature_dict.get("Overtime_Hours", 0)),
                    float(feature_dict.get("Employee_Satisfaction_Score", 0)),
                )

            # Explain AI prediction (ensure we pass scaled inputs matching model training)
            shap_values = explain_prediction(scaled_df) or {}

            emp_id = request.form.get("employee_id") or session.get("employee_id")
            emp_name = request.form.get("employee_name") or session.get("employee_name")
            department = request.form.get("Department") or BASELINE_FEATURE_VALUES.get("Department")

            # Persist employee context so quick mode can remember details
            if emp_id:
                session["employee_id"] = emp_id
            if emp_name:
                session["employee_name"] = emp_name

            # Sort and format the SHAP contributions for display (top contributors)
            shap_sorted = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
            shap_chart = [
                {
                    "feature": k,
                    "impact": round(v, 4),
                    "abs_impact": abs(v),
                    "bar_width": min(abs(v) * 15, 100),
                }
                for k, v in shap_sorted
            ]

            saved = save_prediction(
                emp_id,
                final_score,
                risk,
                employee_name=emp_name,
                department=department,
                monthly_salary=feature_dict.get("Monthly_Salary"),
                projects_handled=feature_dict.get("Projects_Handled"),
                work_hours=feature_dict.get("Work_Hours_Per_Week"),
                satisfaction_score=feature_dict.get("Employee_Satisfaction_Score"),
                team_size=feature_dict.get("Team_Size"),
                years_at_company=feature_dict.get("Years_At_Company"),
                remote_work_frequency=feature_dict.get("Remote_Work_Frequency"),
            )
            emp_id = saved.get("employee_id")
            emp_name = saved.get("employee_name")
            department = saved.get("department")

            metrics = load_metrics()
            regression_metrics = metrics.get("regression", {})
            classification_metrics = metrics.get("classification", {})
            reg_r2 = float(regression_metrics.get("r2_score", 0) or 0)
            reg_rmse = float(regression_metrics.get("rmse", 0) or 0)
            prediction_confidence = round(
                max(0.0, min(100.0, (max(reg_r2, 0.0) * 100 * 0.7) + (max(0.0, 100.0 - reg_rmse) * 0.3))),
                2,
            )

            try:
                log_activity(
                    actor_role="employee",
                    employee_id=emp_id,
                    action="prediction_submitted",
                    path=request.path,
                    method=request.method,
                    metadata=json.dumps(
                        {
                            "prediction": final_score,
                            "risk": risk,
                            "department": department,
                        }
                    ),
                    ip_address=_get_request_ip(),
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception:
                pass

            return render_template(
                "result.html",
                prediction=final_score,
                burnout=risk,
                shap_chart=shap_chart,
                emp_id=emp_id,
                emp_name=emp_name,
                department=department,
                model_metrics=metrics,
                model_accuracy=regression_metrics.get("accuracy_pct"),
                classification_accuracy=classification_metrics.get("accuracy"),
                prediction_confidence=prediction_confidence,
                burnout_confidence=burnout_confidence,
                inputs={
                    "Work Hours / Week": feature_dict.get("Work_Hours_Per_Week"),
                    "Overtime Hours": feature_dict.get("Overtime_Hours"),
                    "Satisfaction Score": feature_dict.get("Employee_Satisfaction_Score"),
                    "Monthly Salary (k)": feature_dict.get("Monthly_Salary"),
                    "Projects Handled": feature_dict.get("Projects_Handled"),
                    "Team Size": feature_dict.get("Team_Size"),
                },
            )

        except Exception as e:
            print("Prediction Error:", e)
            return "Prediction Error: Check form inputs and feature names."

    # GET request: show quick predict form by default
    return render_template("predict_quick.html")


# -------------------------------
# HISTORY
# -------------------------------

@app.route("/history")
@admin_required
def history():

    data = get_history()

    return render_template("history.html", data=data)


# -------------------------------
# DASHBOARD API (LIVE CHARTS)
# -------------------------------

@app.route("/api/dashboard")
def dashboard_api():

    stats = get_dashboard_stats()

    return jsonify(stats)


# -------------------------------
# DEPARTMENT ANALYTICS API
# -------------------------------

@app.route("/api/departments")
def department_api():

    data = department_productivity()

    return jsonify(data)


# -------------------------------
# BURNOUT ALERT API
# -------------------------------

@app.route("/api/alerts")
def alerts_api():

    alerts = burnout_alerts()

    return jsonify(alerts)


# -------------------------------
# PDF REPORT DOWNLOAD
# -------------------------------

@app.route("/download-report/<score>/<risk>")
def download_report(score, risk):

    file = generate_pdf(score, risk)

    return send_file(file, as_attachment=True)


@app.route("/download_report")
def download_report_all():
    """Generate a PDF report including a table of recent predictions."""

    data = get_history(limit=50)

    if data:
        last = data[0]
        # sqlite3.Row does not support .get(); convert to dict to safely access fields
        if hasattr(last, "keys"):
            last = dict(last)
        score = last.get("result", 0)
        risk = last.get("risk", "Unknown")
    else:
        score = 0
        risk = "Unknown"

    file = generate_pdf(score, risk, records=data)

    return send_file(file, as_attachment=True)


# -------------------------------
# LOGOUT
# -------------------------------

@app.route("/logout")
def logout():

    if session.get("employee_id"):
        try:
            log_activity(
                actor_role="employee",
                employee_id=session.get("employee_id"),
                action="logout",
                path=request.path,
                method=request.method,
                ip_address=_get_request_ip(),
                user_agent=request.headers.get("User-Agent"),
            )
        except Exception:
            pass

    session.clear()

    return redirect(url_for("index"))


# -------------------------------
# RUN SERVER
# -------------------------------

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
    # # from flask import Flask, render_template, request, redirect, url_for, send_file
# # import joblib
# # import json
# # import pandas as pd
# # import numpy as np
# # import plotly
# # import plotly.graph_objs as go
# # import matplotlib.pyplot as plt
# # from utils import database as db
# # from utils.risk_detection import detect_risk
# # from utils.pdf_report import generate_pdf_report
# # from utils.burnout_model import burnout_risk
# # from model.eda_engine import generate_eda

# # app = Flask(__name__)
# # generate_eda()
# # # -----------------------------
# # # Load Model Assets
# # # -----------------------------
# # model = joblib.load("model/model.pkl")
# # scaler = joblib.load("model/scaler.pkl")
# # features = joblib.load("model/features.pkl")

# # with open("model/metrics.json") as f:
# #     metrics = json.load(f)

# # db.init_db(features)

# # # -----------------------------
# # # Dashboard
# # # -----------------------------
# # @app.route('/')
# # def index():

# #     gauge = go.Figure(go.Indicator(
# #         mode="gauge+number",
# #         value=metrics['accuracy_pct'],
# #         title={'text': "Model Reliability (%)"},
# #         gauge={
# #             'axis': {'range': [0, 100]},
# #             'bar': {'color': "#3b82f6"},
# #             'steps': [
# #                 {'range': [0,70], 'color': "#fee2e2"},
# #                 {'range': [70,90], 'color': "#fef3c7"},
# #                 {'range': [90,100], 'color': "#dcfce7"}
# #             ]
# #         }
# #     ))

# #     gauge_json = json.dumps(gauge, cls=plotly.utils.PlotlyJSONEncoder)

# #     return render_template(
# #         "index.html",
# #         metrics=metrics,
# #         features=features,
# #         gauge_json=gauge_json
# #     )

# # # -----------------------------
# # # Prediction Modes
# # # -----------------------------
# # @app.route('/predict_mode')
# # def predict_mode():
# #     return render_template("predict_mode.html")

# # @app.route('/predict_quick')
# # def predict_quick():
# #     return render_template("predict_quick.html", features=features[:4])

# # @app.route('/predict_advanced')
# # def predict_advanced():
# #     return render_template(
# #         "predict_advanced.html",
# #         features=features
# #     )

# # # -----------------------------
# # # Run Prediction
# # # -----------------------------
# # @app.route('/run_prediction', methods=['POST'])
# # def run_prediction():

# #     try:

# #         input_data = []

# #         defaults = {
# #             'Age':35,
# #             'Salary':50000,
# #             'Years_at_Company':5,
# #             'Employee_Satisfaction_Score':70,
# #             'Work_Hours_Per_Week':40,
# #             'Overtime_Hours':5
# #         }

# #         for f in features:
# #             val = request.form.get(f)

# #             if val and val.strip() != "":
# #                 input_data.append(float(val))
# #             else:
# #                 input_data.append(defaults.get(f,0))

# #         scaled = scaler.transform([input_data])

# #         prediction = model.predict(scaled)[0]

# #         prediction = float(prediction)

# #         # Ensure valid range
# #         prediction = max(0, min(100, prediction))

# #         prediction = round(prediction, 2)

# #         overtime_val = float(request.form.get("Overtime_Hours",0))

# #         risk_level, risk_msg = detect_risk(
# #             prediction,
# #             1 if overtime_val > 8 else 0
# #         )

# #         burnout = burnout_risk(
# #             float(request.form.get("Work_Hours_Per_Week",40)),
# #             overtime_val,
# #             float(request.form.get("Employee_Satisfaction_Score",70))
# #         )

# #         confidence = round(np.random.uniform(82, 96), 2)
# #         print("MODEL FEATURES:", features)#
# #         db.save_prediction(features, input_data, prediction, risk_level)

# #         # Create feature importance graph
# #         if hasattr(model, "feature_importances_"):

# #             plt.figure(figsize=(8,4))

# #             plt.barh(features, model.feature_importances_)

# #             plt.xlabel("Importance")
# #             plt.title("Model Feature Importance")

# #             plt.tight_layout()

# #             plt.savefig("static/explanation.png")

# #             plt.close()

# #         return render_template(
# #             "result.html",
# #             prediction=prediction,
# #             risk_level=risk_level,
# #             risk_msg=risk_msg,
# #             burnout=burnout,
# #             confidence=confidence,
# #             inputs=dict(zip(features,input_data))
# #         )

# #     except Exception as e:
# #         return f"Prediction Error: {e}",400

# # # insights
# # @app.route('/insights')
# # def insights():

# #     df = pd.read_csv("dataset/employee_data.csv")

# #     dept_productivity = df.groupby("department")["productivity"].mean()

# #     fig = go.Figure(
# #         data=[go.Bar(
# #             x=dept_productivity.index,
# #             y=dept_productivity.values
# #         )]
# #     )

# #     chart = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# #     return render_template(
# #         "performance.html",
# #         chart=chart
# #     )

# # # -----------------------------
# # # History
# # # -----------------------------
# # @app.route('/history')
# # def history():

# #     data = db.get_history()

# #     return render_template("history.html", data=data)


# # # -----------------------------
# # # Trend Chart Data
# # # -----------------------------
# # @app.route("/trend_data")
# # def trend_data():

# #     rows = db.get_history()

# #     ids=[]
# #     preds=[]

# #     for r in rows:
# #         ids.append(r[0])
# #         preds.append(r[-2])

# #     return {"ids":ids,"predictions":preds}


# # # -----------------------------
# # # EDA Dashboard
# # # -----------------------------
# # @app.route('/eda_dashboard')
# # def eda_dashboard():

# #     return render_template("eda_dashboard.html", metrics=metrics)

# # @app.route("/eda_data")
# # def eda_data():

# #     import pandas as pd
# #     import plotly
# #     import plotly.express as px
# #     import json

# #     df = pd.read_csv("dataset/employee_data.csv")

# #     # Heatmap
# #     corr = df.corr(numeric_only=True)

# #     heatmap = px.imshow(
# #         corr,
# #         text_auto=True,
# #         color_continuous_scale="RdBu"
# #     )

# #     # Satisfaction vs productivity
# #     satisfaction = px.scatter(
# #         df,
# #         x="Employee_Satisfaction_Score",
# #         y="Performance_Score",
# #         trendline="ols",
# #         title="Satisfaction vs Productivity"
# #     )

# #     # Salary vs productivity
# #     salary = px.scatter(
# #         df,
# #         x="Salary",
# #         y="Performance_Score",
# #         title="Salary vs Productivity"
# #     )

# #     # Department distribution
# #     dept = px.histogram(
# #         df,
# #         x="Department",
# #         title="Department Distribution"
# #     )

# #     return {
# #         "heatmap": json.loads(plotly.utils.PlotlyJSONEncoder().encode(heatmap)),
# #         "satisfaction": json.loads(plotly.utils.PlotlyJSONEncoder().encode(satisfaction)),
# #         "salary": json.loads(plotly.utils.PlotlyJSONEncoder().encode(salary)),
# #         "dept": json.loads(plotly.utils.PlotlyJSONEncoder().encode(dept))
# #     }
# # @app.route('/refresh_eda')
# # def refresh_eda():

# #     generate_eda()

# #     return redirect(url_for("eda_dashboard"))


# # # -----------------------------
# # # Bulk CSV Prediction
# # # -----------------------------
# # @app.route("/upload")
# # def upload():

# #     return render_template("upload_csv.html")


# # @app.route("/upload_csv", methods=["POST"])
# # def upload_csv():

# #     file = request.files["file"]

# #     df = pd.read_csv(file)

# #     df_scaled = scaler.transform(df[features])

# #     preds = model.predict(df_scaled)

# #     df["Prediction"] = preds

# #     output = "bulk_predictions.csv"

# #     df.to_csv(output,index=False)

# #     return send_file(output, as_attachment=True)


# # # -----------------------------
# # # PDF Report
# # # -----------------------------
# # @app.route("/download_report")
# # def download_report():

# #     data = db.get_history()

# #     file = generate_pdf_report(data)

# #     return send_file(file, as_attachment=True)


# # # -----------------------------
# # # Admin Analytics
# # # -----------------------------
# # @app.route("/performance")
# # def performance():

# #     return render_template(
# #         "performance.html",
# #         metrics=metrics
# #     )


# # # -----------------------------
# # # Run Server
# # # -----------------------------
# # if __name__ == "__main__":
# #     app.run(debug=True)
# from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
# import joblib
# import json
# import pandas as pd
# import numpy as np
# import plotly
# import plotly.graph_objs as go
# import plotly.express as px
# import matplotlib.pyplot as plt

# from utils import database as db
# from utils.risk_detection import detect_risk
# from utils.pdf_report import generate_pdf_report
# from utils.burnout_model import burnout_risk
# from model.eda_engine import generate_eda

# app = Flask(__name__)

# # --------------------------------
# # Load Model
# # --------------------------------
# generate_eda()

# model = joblib.load("model/model.pkl")
# scaler = joblib.load("model/scaler.pkl")
# features = joblib.load("model/features.pkl")

# with open("model/metrics.json") as f:
#     metrics = json.load(f)

# db.init_db(features)

# # --------------------------------
# # DASHBOARD
# # --------------------------------
# @app.route("/")
# def index():

#     gauge = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=metrics['accuracy_pct'],
#         title={'text': "Model Accuracy"},
#         gauge={
#             'axis': {'range': [0, 100]},
#             'bar': {'color': "#3b82f6"},
#             'steps': [
#                 {'range': [0,60], 'color': "#fecaca"},
#                 {'range': [60,80], 'color': "#fde68a"},
#                 {'range': [80,100], 'color': "#bbf7d0"}
#             ]
#         }
#     ))

#     gauge_json = json.dumps(gauge, cls=plotly.utils.PlotlyJSONEncoder)

#     return render_template(
#         "index.html",
#         metrics=metrics,
#         features=features,
#         gauge_json=gauge_json
#     )


# # --------------------------------
# # PREDICTION MODES
# # --------------------------------
# @app.route('/predict_mode')
# def predict_mode():
#     return render_template("predict_mode.html")


# @app.route('/predict_quick')
# def predict_quick():
#     return render_template("predict_quick.html", features=features[:4])


# @app.route('/predict_advanced')
# def predict_advanced():
#     return render_template("predict_advanced.html", features=features)


# # --------------------------------
# # RUN PREDICTION
# # --------------------------------
# @app.route('/run_prediction', methods=['POST'])
# def run_prediction():

#     try:

#         # -----------------------------
#         # Auto Feature Builder
#         # -----------------------------

#         input_data = []

#         department = request.form.get("Department", "")

#         work_hours = float(request.form.get("Work_Hours_Per_Week",40))
#         overtime = float(request.form.get("Overtime_Hours",5))

#         total_effort = work_hours + overtime

#         for f in features:

#             if f == "Total_Effort":
#                 input_data.append(total_effort)

#             elif "Department_" in f:
#                 dept_name = f.split("_")[1]
#                 input_data.append(1 if department == dept_name else 0)

#             else:
#                 val = request.form.get(f)
#                 input_data.append(float(val) if val else 0)

#         scaled = scaler.transform([input_data])

#         prediction = float(model.predict(scaled)[0])
#         prediction = max(0, min(100, prediction))
#         prediction = round(prediction, 2)

#         overtime_val = float(request.form.get("Overtime_Hours",0))

#         risk_level, risk_msg = detect_risk(
#             prediction,
#             1 if overtime_val > 8 else 0
#         )

#         burnout = burnout_risk(
#             float(request.form.get("Work_Hours_Per_Week",40)),
#             overtime_val,
#             float(request.form.get("Employee_Satisfaction_Score",70))
#         )

#         confidence = round(100 - abs(prediction - 75),2)
#         confidence = max(70,min(confidence,98))

#         db.save_prediction(features,input_data,prediction,risk_level)

#         # Feature importance graph
#         if hasattr(model,"feature_importances_"):

#             plt.figure(figsize=(8,4))
#             plt.barh(features,model.feature_importances_)
#             plt.xlabel("Importance")
#             plt.title("Feature Importance")
#             plt.tight_layout()
#             plt.savefig("static/explanation.png")
#             plt.close()

#         return render_template(
#             "result.html",
#             prediction=prediction,
#             risk_level=risk_level,
#             risk_msg=risk_msg,
#             burnout=burnout,
#             confidence=confidence,
#             inputs=dict(zip(features,input_data))
#         )

#     except Exception as e:
#         return f"Prediction Error: {e}",400


# # --------------------------------
# # ANALYTICS PAGE
# # --------------------------------
# @app.route("/performance")
# def performance():
#     return render_template("performance.html", metrics=metrics)


# # --------------------------------
# # ANALYTICS CHART API
# # --------------------------------
# @app.route("/chart/<chart_type>")
# def chart(chart_type):

#     df = pd.read_csv("dataset/employee_data.csv")

#     if chart_type == "department":

#         data = df.groupby("Department")["Performance_Score"].mean()

#         fig = go.Figure(go.Bar(
#             x=data.index,
#             y=data.values
#         ))

#     elif chart_type == "salary":

#         fig = px.scatter(
#             df,
#             x="Monthly_Salary",
#             y="Performance_Score",
#             trendline="ols"
#         )

#     elif chart_type == "satisfaction":

#         fig = px.scatter(
#             df,
#             x="Employee_Satisfaction_Score",
#             y="Performance_Score",
#             trendline="ols"
#         )

#     elif chart_type == "correlation":

#         corr = df.corr(numeric_only=True)

#         fig = px.imshow(corr,text_auto=True)

#     else:

#         fig = go.Figure()

#     return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# # --------------------------------
# # HISTORY
# # --------------------------------
# @app.route('/history')
# def history():

#     data = db.get_history()

#     return render_template("history.html", data=data)


# # --------------------------------
# # BULK CSV PREDICTION
# # --------------------------------
# @app.route("/upload")
# def upload():
#     return render_template("upload_csv.html")


# @app.route("/upload_csv", methods=["POST"])
# def upload_csv():

#     file = request.files["file"]
#     df = pd.read_csv(file)

#     df_scaled = scaler.transform(df[features])

#     preds = model.predict(df_scaled)

#     df["Prediction"] = preds

#     output = "bulk_predictions.csv"

#     df.to_csv(output,index=False)

#     return send_file(output, as_attachment=True)


# # --------------------------------
# # PDF REPORT
# # --------------------------------
# @app.route("/download_report")
# def download_report():

#     data = db.get_history()

#     file = generate_pdf_report(data)

#     return send_file(file, as_attachment=True)


# # --------------------------------
# # SERVER
# # --------------------------------
# if __name__ == "__main__":
#     app.run(debug=True)
