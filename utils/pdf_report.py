from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def generate_pdf(score, risk, records=None):

    file="report.pdf"

    doc = SimpleDocTemplate(file, pagesize=letter)

    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Employee Productivity Report",styles["Title"]))

    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Predicted Productivity: {score}", styles["Normal"]))
    content.append(Paragraph(f"Burnout Risk: {risk}", styles["Normal"]))
    content.append(Spacer(1, 16))

    if records:
        content.append(Paragraph("Recent Audit Records", styles["Heading2"]))
        content.append(Spacer(1, 12))

        table_data = [["ID", "Employee", "Department", "Score", "Risk"]]
        for r in records:
            table_data.append(
                [
                    str(r["id"] if "id" in r.keys() else ""),
                    str(r["employee_id"] if "employee_id" in r.keys() else ""),
                    str(r["department"] if "department" in r.keys() else ""),
                    str(r["result"] if "result" in r.keys() else ""),
                    str(r["risk"] if "risk" in r.keys() else ""),
                ]
            )

        table = Table(table_data, colWidths=[40, 90, 90, 70, 100])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        content.append(table)

    doc.build(content)

    return file