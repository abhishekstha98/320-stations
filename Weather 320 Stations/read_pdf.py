
import sys
import os

try:
    from pypdf import PdfReader
except ImportError:
    try:
        import PyPDF2 as PdfReader
    except ImportError:
        print("Neither pypdf nor PyPDF2 is installed.")
        # Try a simpler raw string search if libs are missing, just in case
        sys.exit(0)

pdf_path = r"c:/Users/abhis/Downloads/Weather 320 Stations-20260111T031707Z-1-001/Weather 320 Stations/thesis_final.pdf"

try:
    reader = PdfReader(pdf_path)
    # Print first 5 pages to ensure we catch the title page and abstract
    for i in range(min(5, len(reader.pages))):
        print(f"--- Page {i+1} ---")
        print(reader.pages[i].extract_text())
except Exception as e:
    print(f"Error reading PDF: {e}")
