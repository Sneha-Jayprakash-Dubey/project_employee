from utils.database import get_department_averages, get_recent_predictions, get_db_connection


def department_productivity():
    """Return average productivity by department."""

    rows = get_department_averages()
    return [{"department": r["department"], "avg_score": r["avg_score"]} for r in rows]


def burnout_alerts(limit=20):
    """Return recent employees with high burnout risk."""

    rows = get_recent_predictions(limit)
    alerts = []
    for r in rows:
        if r.get("risk", "").lower().startswith("high"):
            alerts.append({"employee_id": r.get("employee_id"), "burnout_risk": r.get("risk")})
    return alerts


def burnout_alerts(limit=20):
    """Return recent employee IDs flagged with high burnout risk."""

    conn = get_db_connection()
    rows = conn.execute(
        "SELECT employee_id, risk FROM predictions WHERE risk LIKE 'High%' ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    return [{"employee_id": r["employee_id"], "burnout_risk": r["risk"]} for r in rows]