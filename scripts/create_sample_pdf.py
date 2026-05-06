"""
Create a sample employee handbook PDF for testing.
Uses reportlab to generate a simple 2-page PDF.
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from pathlib import Path


def create_sample_handbook():
    """Create sample employee handbook PDF."""
    output_path = Path(__file__).parent.parent / "data" / "sample_handbook.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    
    # Page 1: Vacation Policy
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, height - 1*inch, "Employee Handbook")
    
    c.setFont("Helvetica-Bold", 14)
    y = height - 1.5*inch
    c.drawString(1*inch, y, "Chapter 1: Vacation Policy")
    
    c.setFont("Helvetica", 11)
    y -= 0.4*inch
    content_page1 = [
        "Employees are entitled to 12 days of paid vacation per year.",
        "Vacation time must be requested at least 2 weeks in advance.",
        "Manager approval is required before booking vacation.",
        "Unused vacation days do not carry over to the next year.",
        "During vacation, employees are expected to be unreachable.",
        "",
        "Vacation Request Process:",
        "1. Submit request to your manager via email",
        "2. Wait for manager approval (typically 1-3 business days)",
        "3. Once approved, confirm with HR department",
        "4. Update your calendar and notify your team",
    ]
    
    for line in content_page1:
        c.drawString(1*inch, y, line)
        y -= 0.25*inch
    
    # Page 2: Sick Leave
    c.showPage()
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, height - 1*inch, "Employee Handbook (continued)")
    
    c.setFont("Helvetica-Bold", 14)
    y = height - 1.5*inch
    c.drawString(1*inch, y, "Chapter 2: Sick Leave")
    
    c.setFont("Helvetica", 11)
    y -= 0.4*inch
    content_page2 = [
        "Sick leave: up to 10 days per year for personal illness.",
        "For absences of 3 or more consecutive days, a doctor's note is required.",
        "Sick leave must be reported to your manager as soon as possible.",
        "Sick leave can be taken in half-day increments.",
        "",
        "Medical Certificate Requirements:",
        "- Required for absences longer than 3 working days",
        "- Must be submitted within 48 hours of return",
        "- Certificate must clearly state dates of medical visit",
        "",
        "Chapter 3: Overtime Policy",
        "Overtime is compensated at 1.5x regular hourly rate.",
        "All overtime must be pre-approved by management.",
        "Maximum 10 hours overtime per week to prevent burnout.",
    ]
    
    for line in content_page2:
        c.drawString(1*inch, y, line)
        y -= 0.25*inch
    
    c.save()
    print(f"✓ Created sample handbook: {output_path}")


if __name__ == "__main__":
    create_sample_handbook()
