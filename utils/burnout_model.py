def burnout_risk(hours, overtime, satisfaction):
    """Estimate burnout risk using normalized workload, overtime, and satisfaction."""

    try:
        hours = float(hours)
    except (TypeError, ValueError):
        hours = 0.0
    try:
        overtime = float(overtime)
    except (TypeError, ValueError):
        overtime = 0.0
    try:
        satisfaction = float(satisfaction)
    except (TypeError, ValueError):
        satisfaction = 0.0

    # Accept either 1-5 or 0-100 satisfaction scales.
    if satisfaction <= 5:
        satisfaction = satisfaction * 20.0
    satisfaction = max(0.0, min(100.0, satisfaction))

    hours_risk = max(0.0, min(100.0, ((hours - 30.0) / 30.0) * 100.0))
    overtime_risk = max(0.0, min(100.0, (overtime / 20.0) * 100.0))
    satisfaction_risk = 100.0 - satisfaction

    score = (0.45 * hours_risk) + (0.35 * overtime_risk) + (0.20 * satisfaction_risk)

    if score >= 65:
        return "High Burnout Risk"
    if score >= 40:
        return "Moderate Burnout Risk"
    return "Low Burnout Risk"
