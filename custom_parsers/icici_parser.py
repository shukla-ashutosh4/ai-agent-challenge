import PyPDF2
import pandas as pd
import re
from typing import List, Dict

def parse(pdf_path: str) -> pd.DataFrame:
    try:
        with open(pdf_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
        
        data = []
        pattern = r'(?P<Date>\d{2}-\d{2}-\d{4})\s+(?P<Description>.*)\s+(?P<Debit_Amt>\d*\.?\d*)\s+(?P<Credit_Amt>\d*\.?\d*)\s+(?P<Balance>\d*\.?\d*)'
        matches = re.findall(pattern, text)
        
        for match in matches:
            date, description, debit_amt, credit_amt, balance = match
            data.append({'Date': date, 'Description': description, 
                        'Debit Amt': float(debit_amt) if debit_amt else None, 
                        'Credit Amt': float(credit_amt) if credit_amt else None, 
                        'Balance': float(balance)})
        
        columns = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
        df = pd.DataFrame(data)
        
        return df[columns]
    
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return None