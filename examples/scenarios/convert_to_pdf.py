#!/usr/bin/env python3
"""
Convert all scenario text files to PDF format.
Usage: python convert_to_pdf.py
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfgen import canvas

def create_pdf_from_text(text_file: Path, pdf_file: Path):
    """Convert a text file to PDF with proper formatting"""
    
    # Read the text file
    with open(text_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create PDF
    doc = SimpleDocTemplate(
        str(pdf_file),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor='#1a1a1a',
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor='#333333',
        spaceAfter=6,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        textColor='#000000',
        spaceAfter=6,
        alignment=TA_LEFT,
        fontName='Helvetica'
    )
    
    separator_style = ParagraphStyle(
        'Separator',
        parent=styles['Normal'],
        fontSize=10,
        textColor='#666666',
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Courier'
    )
    
    # Process content line by line
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if not line:
            elements.append(Spacer(1, 0.1*inch))
            continue
        
        # Check for separator lines (====)
        if line.startswith('===='):
            elements.append(Paragraph(line, separator_style))
            continue
        
        # Check for section headers (ALL CAPS followed by colon)
        if line.isupper() and ':' in line and len(line) < 50:
            elements.append(Paragraph(f"<b>{line}</b>", heading_style))
            continue
        
        # Check for field labels (Field Name: Value)
        if ':' in line and not line.startswith(' ') and len(line.split(':')[0]) < 40:
            parts = line.split(':', 1)
            if len(parts) == 2:
                field_name = parts[0].strip()
                field_value = parts[1].strip()
                formatted_line = f"<b>{field_name}:</b> {field_value}"
                elements.append(Paragraph(formatted_line, body_style))
                continue
        
        # Check for bullet points or checkmarks
        if line.startswith(('✓', '✗', '⚠️', '🚨', '-', '•')):
            elements.append(Paragraph(line, body_style))
            continue
        
        # Regular text
        elements.append(Paragraph(line, body_style))
    
    # Build PDF
    doc.build(elements)
    print(f"✓ Created: {pdf_file.name}")


def main():
    """Convert all text files in scenarios to PDF"""
    scenarios_dir = Path(__file__).parent
    
    print("Converting scenario text files to PDF...\n")
    
    # Find all .txt files
    txt_files = list(scenarios_dir.rglob("*.txt"))
    
    if not txt_files:
        print("No text files found!")
        return
    
    converted = 0
    for txt_file in txt_files:
        # Skip if it's not a claim document
        if txt_file.name == "README.txt":
            continue
        
        # Create PDF filename
        pdf_file = txt_file.with_suffix('.pdf')
        
        try:
            create_pdf_from_text(txt_file, pdf_file)
            converted += 1
        except Exception as e:
            print(f"✗ Failed to convert {txt_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete: {converted}/{len(txt_files)} files converted")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
