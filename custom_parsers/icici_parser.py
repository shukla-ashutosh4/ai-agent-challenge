import pdfplumber
import pandas as pd
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def parse(pdf_path: str) -> pd.DataFrame:
    """Parse ICICI bank statement PDF."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

        pattern = r"(\d{2}-\d{2}-\d{4})\s+([\w\s]+)\s*(\d+\.\d+|\s*)\s*(\d+\.\d+|\s*)\s*(\d+\.\d+)"
        matches = re.findall(pattern, text)

        data = []
        for match in matches:
            date, description, debit, credit, balance = match
            data.append({
                "Date": date,
                "Description": description,
                "Debit Amt": float(debit) if debit else None,
                "Credit Amt": float(credit) if credit else None,
                "Balance": float(balance)
            })

        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])