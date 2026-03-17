import sqlite3
import os

# Define database path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "..", "database.db"))


def ensure_schema(conn):
    """Ensure the predictions table has all expected columns.

    This allows adding new columns (like created_at) without dropping existing data.
    """

    cur = conn.cursor()

    # Ensure the predictions table exists before making schema changes.
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
    )
    if cur.fetchone() is None:
        return

    # Add created_at column if missing so we can filter by time range.
    cur.execute("PRAGMA table_info(predictions)")
    cols = [r[1] for r in cur.fetchall()]

    if "created_at" not in cols:
        # SQLite does not allow non-constant default expressions, so add the column without a default
        cur.execute("ALTER TABLE predictions ADD COLUMN created_at TEXT")
        # Backfill existing rows (keep existing values if present)
        cur.execute(
            "UPDATE predictions SET created_at = datetime('now') WHERE created_at IS NULL"
        )
        conn.commit()

    # Add optional analytics columns if missing.
    for col_name in ("team_size", "years_at_company", "remote_work_frequency"):
        if col_name not in cols:
            cur.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} REAL")
            conn.commit()

    # Canonical employee profile table to prevent ID/name/department mismatches.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            employee_name TEXT,
            department TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()

    # Admin-configured productivity goals per department.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS department_goals (
            department TEXT PRIMARY KEY,
            target REAL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()

    # Activity log for auditing employee/admin actions.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor_role TEXT,
            employee_id TEXT,
            action TEXT,
            path TEXT,
            method TEXT,
            metadata TEXT,
            ip_address TEXT,
            user_agent TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()

    # Build/refresh canonical employee profiles from latest prediction rows.
    cur.execute(
        """
        INSERT INTO employees (employee_id, employee_name, department, updated_at)
        SELECT p.employee_id, p.employee_name, p.department, CURRENT_TIMESTAMP
        FROM predictions p
        JOIN (
            SELECT employee_id, MAX(id) AS max_id
            FROM predictions
            WHERE employee_id IS NOT NULL AND TRIM(employee_id) <> ''
            GROUP BY employee_id
        ) latest
            ON p.employee_id = latest.employee_id AND p.id = latest.max_id
        ON CONFLICT(employee_id) DO UPDATE SET
            employee_name = COALESCE(excluded.employee_name, employees.employee_name),
            department = COALESCE(excluded.department, employees.department),
            updated_at = CURRENT_TIMESTAMP
        """
    )
    conn.commit()

    # Reconcile existing prediction rows to canonical profiles.
    cur.execute(
        """
        UPDATE predictions
        SET employee_name = COALESCE(
                (SELECT e.employee_name FROM employees e WHERE e.employee_id = predictions.employee_id),
                employee_name
            ),
            department = COALESCE(
                (SELECT e.department FROM employees e WHERE e.employee_id = predictions.employee_id),
                department
            )
        WHERE employee_id IS NOT NULL AND TRIM(employee_id) <> ''
        """
    )
    conn.commit()


def normalize_text(value):
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def resolve_employee_profile(conn, employee_id, employee_name=None, department=None):
    """Return a canonical (name, department) for a given employee ID."""

    employee_id = normalize_text(employee_id)
    employee_name = normalize_text(employee_name)
    department = normalize_text(department)

    if not employee_id:
        return employee_id, employee_name, department

    row = conn.execute(
        "SELECT employee_name, department FROM employees WHERE employee_id = ?",
        (employee_id,),
    ).fetchone()

    if row:
        canonical_name = row["employee_name"] or employee_name
        canonical_department = row["department"] or department

        # Backfill canonical profile if DB profile had missing values.
        if (row["employee_name"] is None and employee_name) or (row["department"] is None and department):
            conn.execute(
                """
                UPDATE employees
                SET employee_name = COALESCE(employee_name, ?),
                    department = COALESCE(department, ?),
                    updated_at = CURRENT_TIMESTAMP
                WHERE employee_id = ?
                """,
                (employee_name, department, employee_id),
            )
            conn.commit()

        return employee_id, canonical_name, canonical_department

    conn.execute(
        """
        INSERT INTO employees (employee_id, employee_name, department, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (employee_id, employee_name, department),
    )
    conn.commit()
    return employee_id, employee_name, department


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Ensure the schema is up-to-date on every connection
    try:
        ensure_schema(conn)
    except Exception:
        # If schema enforce fails for any reason, continue with the connection.
        pass
    return conn


def _apply_segment_filters(
    where_clauses,
    params,
    team_size_band=None,
    experience_band=None,
    salary_band=None,
    remote_band=None,
):
    if team_size_band == "Small":
        where_clauses.append("team_size IS NOT NULL AND team_size <= 5")
    elif team_size_band == "Medium":
        where_clauses.append("team_size IS NOT NULL AND team_size BETWEEN 6 AND 12")
    elif team_size_band == "Large":
        where_clauses.append("team_size IS NOT NULL AND team_size > 12")

    if experience_band == "Junior":
        where_clauses.append("years_at_company IS NOT NULL AND years_at_company <= 2")
    elif experience_band == "Mid":
        where_clauses.append("years_at_company IS NOT NULL AND years_at_company BETWEEN 3 AND 6")
    elif experience_band == "Senior":
        where_clauses.append("years_at_company IS NOT NULL AND years_at_company BETWEEN 7 AND 12")
    elif experience_band == "Veteran":
        where_clauses.append("years_at_company IS NOT NULL AND years_at_company > 12")

    if salary_band == "Low":
        where_clauses.append("monthly_salary IS NOT NULL AND monthly_salary < 40")
    elif salary_band == "Mid":
        where_clauses.append("monthly_salary IS NOT NULL AND monthly_salary BETWEEN 40 AND 80")
    elif salary_band == "High":
        where_clauses.append("monthly_salary IS NOT NULL AND monthly_salary > 80")

    if remote_band == "Onsite":
        where_clauses.append("remote_work_frequency IS NOT NULL AND remote_work_frequency <= 1")
    elif remote_band == "Hybrid":
        where_clauses.append("remote_work_frequency IS NOT NULL AND remote_work_frequency BETWEEN 2 AND 3")
    elif remote_band == "Remote":
        where_clauses.append("remote_work_frequency IS NOT NULL AND remote_work_frequency >= 4")

    return where_clauses, params

# Ensure this name matches 'save_prediction' exactly
def save_prediction(
    employee_id,
    result,
    risk,
    employee_name=None,
    department=None,
    monthly_salary=None,
    projects_handled=None,
    work_hours=None,
    satisfaction_score=None,
    team_size=None,
    years_at_company=None,
    remote_work_frequency=None,
    created_at=None,
):
    """Save a prediction record to the database.

    This schema supports richer record fields, but only employee_id/result/risk are required.
    """

    conn = get_db_connection()

    if created_at is None:
        from datetime import datetime

        created_at = datetime.now().isoformat(sep=' ')

    employee_id, employee_name, department = resolve_employee_profile(
        conn, employee_id, employee_name, department
    )

    query = """
        INSERT INTO predictions (
            employee_id,
            employee_name,
            department,
            monthly_salary,
            projects_handled,
            work_hours,
            satisfaction_score,
            result,
            risk,
            created_at,
            team_size,
            years_at_company,
            remote_work_frequency
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    conn.execute(
        query,
        (
            employee_id,
            employee_name,
            department,
            monthly_salary,
            projects_handled,
            work_hours,
            satisfaction_score,
            round(result, 2) if result is not None else None,
            risk,
            created_at,
            team_size,
            years_at_company,
            remote_work_frequency,
        ),
    )
    conn.commit()
    conn.close()
    return {
        "employee_id": employee_id,
        "employee_name": employee_name,
        "department": department,
    }

# Ensure this name matches 'get_history' exactly
def get_history(
    emp_id=None,
    limit=50,
    days=None,
    department=None,
    risk=None,
    team_size_band=None,
    experience_band=None,
    salary_band=None,
    remote_band=None,
):
    conn = get_db_connection()

    where_clauses = []
    params = []

    if emp_id:
        where_clauses.append("employee_id = ?")
        params.append(emp_id)

    if days is not None:
        where_clauses.append("created_at >= datetime('now', ?)")
        params.append(f"-{int(days)} days")

    if department:
        where_clauses.append("department = ?")
        params.append(department)

    if risk:
        if risk == "High":
            where_clauses.append("risk LIKE 'High%'")
        elif risk == "Moderate":
            where_clauses.append("risk LIKE 'Moderate%'")
        elif risk == "Low":
            where_clauses.append("risk NOT LIKE 'High%' AND risk NOT LIKE 'Moderate%'")

    _apply_segment_filters(
        where_clauses,
        params,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
    )

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    query = f"SELECT * FROM predictions {where_sql} ORDER BY id DESC LIMIT ?"
    params.append(limit)

    data = conn.execute(query, tuple(params)).fetchall()
    conn.close()
    return data


def get_recent_predictions(
    limit=20,
    days=None,
    emp_id=None,
    department=None,
    risk=None,
    team_size_band=None,
    experience_band=None,
    salary_band=None,
    remote_band=None,
):
    """Return the most recent predictions (for trend charts)."""

    conn = get_db_connection()

    where_clauses = []
    params = []

    if emp_id:
        where_clauses.append("employee_id = ?")
        params.append(emp_id)

    if days is not None:
        where_clauses.append("created_at >= datetime('now', ?)")
        params.append(f"-{int(days)} days")

    if department:
        where_clauses.append("department = ?")
        params.append(department)

    if risk:
        if risk == "High":
            where_clauses.append("risk LIKE 'High%'")
        elif risk == "Moderate":
            where_clauses.append("risk LIKE 'Moderate%'")
        elif risk == "Low":
            where_clauses.append("risk NOT LIKE 'High%' AND risk NOT LIKE 'Moderate%'")

    _apply_segment_filters(
        where_clauses,
        params,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
    )

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    query = f"SELECT id, employee_id, result, risk FROM predictions {where_sql} ORDER BY id DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, tuple(params)).fetchall()
    conn.close()
    return rows


def get_department_averages(
    days=None,
    emp_id=None,
    department=None,
    risk=None,
    team_size_band=None,
    experience_band=None,
    salary_band=None,
    remote_band=None,
):
    """Return average result grouped by department."""

    conn = get_db_connection()

    where_clauses = []
    params = []

    if emp_id:
        where_clauses.append("employee_id = ?")
        params.append(emp_id)

    if days is not None:
        where_clauses.append("created_at >= datetime('now', ?)")
        params.append(f"-{int(days)} days")

    if department:
        where_clauses.append("department = ?")
        params.append(department)

    if risk:
        if risk == "High":
            where_clauses.append("risk LIKE 'High%'")
        elif risk == "Moderate":
            where_clauses.append("risk LIKE 'Moderate%'")
        elif risk == "Low":
            where_clauses.append("risk NOT LIKE 'High%' AND risk NOT LIKE 'Moderate%'")

    _apply_segment_filters(
        where_clauses,
        params,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
    )

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    query = f"SELECT department, AVG(result) as avg_score FROM predictions {where_sql} GROUP BY department"
    rows = conn.execute(query, tuple(params)).fetchall()
    conn.close()
    return rows


def get_risk_counts(
    days=None,
    emp_id=None,
    department=None,
    risk=None,
    team_size_band=None,
    experience_band=None,
    salary_band=None,
    remote_band=None,
):
    """Return counts of predictions by risk category."""

    conn = get_db_connection()

    where_clauses = []
    params = []

    if emp_id:
        where_clauses.append("employee_id = ?")
        params.append(emp_id)

    if days is not None:
        where_clauses.append("created_at >= datetime('now', ?)")
        params.append(f"-{int(days)} days")

    if department:
        where_clauses.append("department = ?")
        params.append(department)

    if risk:
        if risk == "High":
            where_clauses.append("risk LIKE 'High%'")
        elif risk == "Moderate":
            where_clauses.append("risk LIKE 'Moderate%'")
        elif risk == "Low":
            where_clauses.append("risk NOT LIKE 'High%' AND risk NOT LIKE 'Moderate%'")

    _apply_segment_filters(
        where_clauses,
        params,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
    )

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    query = f"""
        SELECT
            SUM(CASE WHEN risk LIKE 'High%' THEN 1 ELSE 0 END) AS high,
            SUM(CASE WHEN risk LIKE 'Moderate%' THEN 1 ELSE 0 END) AS moderate,
            SUM(CASE WHEN risk NOT LIKE 'High%' AND risk NOT LIKE 'Moderate%' THEN 1 ELSE 0 END) AS low
        FROM predictions
        {where_sql}
        """

    row = conn.execute(query, tuple(params)).fetchone()
    conn.close()
    return {
        "high": row["high"] or 0,
        "moderate": row["moderate"] or 0,
        "low": row["low"] or 0,
    }


def get_dashboard_stats(
    days=None,
    emp_id=None,
    department=None,
    risk=None,
    team_size_band=None,
    experience_band=None,
    salary_band=None,
    remote_band=None,
):
    conn = get_db_connection()

    where_clauses = []
    params = []

    if emp_id:
        where_clauses.append("employee_id = ?")
        params.append(emp_id)

    if days is not None:
        where_clauses.append("created_at >= datetime('now', ?)")
        params.append(f"-{int(days)} days")

    if department:
        where_clauses.append("department = ?")
        params.append(department)

    if risk:
        if risk == "High":
            where_clauses.append("risk LIKE 'High%'")
        elif risk == "Moderate":
            where_clauses.append("risk LIKE 'Moderate%'")
        elif risk == "Low":
            where_clauses.append("risk NOT LIKE 'High%' AND risk NOT LIKE 'Moderate%'")

    _apply_segment_filters(
        where_clauses,
        params,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
    )

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    query = f"""
        SELECT
            COUNT(*) AS total,
            AVG(result) AS avg_score,
            COALESCE(SUM(CASE WHEN risk LIKE 'High%' THEN 1 ELSE 0 END), 0) AS high_risk_count,
            COALESCE(SUM(CASE WHEN risk LIKE 'Moderate%' THEN 1 ELSE 0 END), 0) AS moderate_risk_count
        FROM predictions
        {where_sql}
        """

    row = conn.execute(query, tuple(params)).fetchone()
    conn.close()

    return {
        "total_predictions": row["total"],
        "avg_score": round(row["avg_score"], 2) if row["avg_score"] is not None else 0,
        "high_risk_count": row["high_risk_count"],
        "moderate_risk_count": row["moderate_risk_count"],
    }


def get_unique_employee_count(days=None):
    """Return count of unique employee IDs in the filtered dataset."""

    conn = get_db_connection()

    where_clauses = []
    params = []
    if days is not None:
        where_clauses.append("created_at >= datetime('now', ?)")
        params.append(f"-{int(days)} days")

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    query = f"SELECT COUNT(DISTINCT employee_id) AS unique_count FROM predictions {where_sql}"

    row = conn.execute(query, tuple(params)).fetchone()
    conn.close()
    return row["unique_count"]


def get_last_prediction_datetime(days=None, emp_id=None):
    """Return the timestamp of the most recent prediction in the filtered dataset."""

    conn = get_db_connection()

    where_clauses = []
    params = []
    if emp_id:
        where_clauses.append("employee_id = ?")
        params.append(emp_id)

    if days is not None:
        where_clauses.append("created_at >= datetime('now', ?)")
        params.append(f"-{int(days)} days")

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    query = f"SELECT MAX(created_at) AS last_seen FROM predictions {where_sql}"

    row = conn.execute(query, tuple(params)).fetchone()
    conn.close()
    return row["last_seen"] if row else None


def burnout_alerts(limit=20, days=None):
    conn = get_db_connection()

    where_clauses = ["risk LIKE 'High%'" ]
    params = []

    if days is not None:
        where_clauses.append("created_at >= datetime('now', ?)")
        params.append(f"-{int(days)} days")

    where_sql = "WHERE " + " AND ".join(where_clauses)

    query = f"SELECT employee_id, risk FROM predictions {where_sql} ORDER BY id DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, tuple(params)).fetchall()
    conn.close()

    return [
        {"employee_id": r["employee_id"], "burnout_risk": r["risk"]} for r in rows
    ]


def get_department_goals():
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT department, target FROM department_goals ORDER BY department"
    ).fetchall()
    conn.close()
    return {r["department"]: r["target"] for r in rows}


def upsert_department_goal(department, target):
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO department_goals (department, target, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(department) DO UPDATE SET
            target = excluded.target,
            updated_at = CURRENT_TIMESTAMP
        """,
        (department, target),
    )
    conn.commit()
    conn.close()


def get_employee_summary(
    days=None,
    department=None,
    risk=None,
    team_size_band=None,
    experience_band=None,
    salary_band=None,
    remote_band=None,
):
    conn = get_db_connection()

    where_clauses = ["employee_id IS NOT NULL", "TRIM(employee_id) <> ''"]
    params = []

    if days is not None:
        where_clauses.append("created_at >= datetime('now', ?)")
        params.append(f"-{int(days)} days")

    if department:
        where_clauses.append("department = ?")
        params.append(department)

    if risk:
        if risk == "High":
            where_clauses.append("risk LIKE 'High%'")
        elif risk == "Moderate":
            where_clauses.append("risk LIKE 'Moderate%'")
        elif risk == "Low":
            where_clauses.append("risk NOT LIKE 'High%' AND risk NOT LIKE 'Moderate%'")

    _apply_segment_filters(
        where_clauses,
        params,
        team_size_band=team_size_band,
        experience_band=experience_band,
        salary_band=salary_band,
        remote_band=remote_band,
    )

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    query = f"""
        SELECT
            p.employee_id,
            p.employee_name,
            p.department,
            AVG(p.result) AS avg_score,
            MAX(p.created_at) AS last_seen,
            (
                SELECT risk
                FROM predictions p2
                WHERE p2.employee_id = p.employee_id
                ORDER BY p2.id DESC
                LIMIT 1
            ) AS last_risk
        FROM predictions p
        {where_sql}
        GROUP BY p.employee_id, p.employee_name, p.department
        ORDER BY avg_score DESC
    """

    rows = conn.execute(query, tuple(params)).fetchall()
    conn.close()
    return rows


def log_activity(
    actor_role,
    employee_id,
    action,
    path=None,
    method=None,
    metadata=None,
    ip_address=None,
    user_agent=None,
):
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO activity_log (
            actor_role,
            employee_id,
            action,
            path,
            method,
            metadata,
            ip_address,
            user_agent,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (
            actor_role,
            employee_id,
            action,
            path,
            method,
            metadata,
            ip_address,
            user_agent,
        ),
    )
    conn.commit()
    conn.close()


def get_activity_log(employee_id=None, days=None, action=None, limit=200):
    conn = get_db_connection()
    where_clauses = []
    params = []

    if employee_id:
        where_clauses.append("employee_id = ?")
        params.append(employee_id)

    if action:
        where_clauses.append("action = ?")
        params.append(action)

    if days is not None:
        where_clauses.append("created_at >= datetime('now', ?)")
        params.append(f"-{int(days)} days")

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    query = f"""
        SELECT *
        FROM activity_log
        {where_sql}
        ORDER BY id DESC
        LIMIT ?
    """
    params.append(limit)
    rows = conn.execute(query, tuple(params)).fetchall()
    conn.close()
    return rows
