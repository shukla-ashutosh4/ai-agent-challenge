#!/usr/bin/env python3
"""
Simplified AI Agent for PDF Parser Generation (Groq version)
A more straightforward version for learning and debugging
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
import time
import traceback

# PDF processing
import PyPDF2
import pandas as pd

# Groq AI client
from groq import Groq
import groq as groq_sdk  # for exceptions and types if needed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimplePDFParserAgent:
    """
    Simplified version of the PDF parser agent (Groq client)
    Easier to understand and debug
    """

    def __init__(self, api_key: str, model: str | None = None):
        """Initialize the agent"""
        if not api_key:
            raise ValueError("API key is required to initialize Groq client")
        self.client = Groq(api_key=api_key)
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.max_attempts = 3

    def read_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text[:2000]  # First 2000 chars for analysis
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return ""

    def read_csv_structure(self, csv_path: str) -> dict:
        """Analyze expected CSV structure"""
        try:
            df = pd.read_csv(csv_path)
            return {
                "columns": list(df.columns),
                "sample_data": df.head(3).to_dict("records"),
                "shape": df.shape,
            }
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return {}

    def _call_groq(self, prompt: str, max_tokens: int = 8192) -> str:
        """Call Groq chat completions API and return assistant content text"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert Python developer."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            # Pull assistant text
            # The Groq python client uses response.choices[0].message.content in the README
            content = ""
            if hasattr(response, "choices") and len(response.choices) > 0:
                # Some responses nest differently — try to extract defensively
                choice = response.choices[0]
                if hasattr(choice, "message") and getattr(choice.message, "content", None) is not None:
                    content = choice.message.content
                elif getattr(choice, "text", None) is not None:
                    content = choice.text
                else:
                    # Fallback to stringifying the response
                    content = str(response)
            else:
                content = str(response)
            return content or ""
        except groq_sdk.APIError as e:
            logger.error(f"Groq API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Groq: {e}")
            raise

    def generate_parser_code(self, bank_name: str, pdf_text: str, csv_structure: dict) -> str:
        """Generate parser code using Groq LLM"""
        prompt = f"""
You are an expert Python developer. Create a PDF parser for {bank_name} bank statements.

PDF SAMPLE TEXT:
```
{pdf_text}
```

REQUIRED OUTPUT STRUCTURE:
- Columns: {csv_structure.get('columns', [])}
- Sample data: {json.dumps(csv_structure.get('sample_data', []), indent=2)}

Create a complete Python function with this EXACT signature:

```python
import PyPDF2
import pandas as pd
import re
from typing import List, Dict

def parse(pdf_path: str) -> pd.DataFrame:
    \"\"\"
    Parse {bank_name} bank statement PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        DataFrame with columns: {csv_structure.get('columns', [])}
    \"\"\"
    # Your implementation here
    pass
```

REQUIREMENTS:
1. Use PyPDF2 or pdfplumber to read the PDF
2. Use regex patterns or table extraction to extract data
3. Return a pandas DataFrame with the exact columns specified
4. Handle errors gracefully
5. The function MUST be named 'parse'

Generate ONLY the complete working Python code. No explanations.
"""
        try:
            logger.debug("Calling Groq to generate parser code...")
            code = self._call_groq(prompt)
            # Clean up markdown formatting if present
            if "```python" in code:
                code = code.split("```python", 1)[1].split("```", 1)[0]
            elif "```" in code:
                code = code.split("```", 1)[1].split("```", 1)[0]
            return code.strip()
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return ""

    def save_parser(self, code: str, output_path: str) -> bool:
        """Save generated parser code"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(code)
            logger.info(f"Saved parser to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving parser: {e}")
            return False

    def test_parser(self, parser_path: str, pdf_path: str, csv_path: str) -> dict:
        """Test the generated parser"""
        result = {"success": False, "error": None, "matches_expected": False}

        try:
            # Import the parser
            sys.path.insert(0, os.path.dirname(parser_path))
            module_name = Path(parser_path).stem

            # Remove from cache if exists
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Import and test
            parser_module = __import__(module_name)

            # Run parser
            parsed_df = parser_module.parse(pdf_path)

            # Load expected data
            expected_df = pd.read_csv(csv_path)

            # Check results
            if isinstance(parsed_df, pd.DataFrame):
                if list(parsed_df.columns) == list(expected_df.columns):
                    result["success"] = True
                    if parsed_df.shape == expected_df.shape:
                        result["matches_expected"] = True
                else:
                    result[
                        "error"
                    ] = f"Column mismatch. Got: {list(parsed_df.columns)}, Expected: {list(expected_df.columns)}"
            else:
                result["error"] = "Parser did not return a DataFrame"

        except Exception as e:
            result["error"] = f"Test failed: {str(e)}\n{traceback.format_exc()}"

        return result

    def fix_parser_code(self, original_code: str, error_message: str) -> str:
        """Generate fixed parser code using Groq"""
        prompt = f"""
The following Python parser code failed with an error:

ERROR: {error_message}

ORIGINAL CODE:
```python
{original_code}
```

Please fix the code to resolve this error. Return ONLY the corrected Python code without any markdown formatting.

Requirements:
1. Keep the same function signature: def parse(pdf_path: str) -> pd.DataFrame
2. Fix the specific error mentioned
3. Ensure robust error handling
4. Return complete working code
"""
        try:
            logger.debug("Calling Groq to fix parser code...")
            fixed = self._call_groq(prompt)
            # Clean up markdown if present
            if "```python" in fixed:
                fixed = fixed.split("```python", 1)[1].split("```", 1)[0]
            elif "```" in fixed:
                fixed = fixed.split("```", 1)[1].split("```", 1)[0]
            return fixed.strip()
        except Exception as e:
            logger.error(f"Error fixing code: {e}")
            return original_code

    def run_agent(self, bank_name: str, pdf_path: str, csv_path: str) -> bool:
        """
        Main agent loop: Generate → Test → Fix → Repeat
        """

        logger.info(f"🚀 Starting agent for {bank_name} parser...")

        # Validate inputs
        if not os.path.exists(pdf_path):
            logger.error(f"PDF not found: {pdf_path}")
            return False

        if not os.path.exists(csv_path):
            logger.error(f"CSV not found: {csv_path}")
            return False

        # Analyze input data
        logger.info("📖 Analyzing PDF and CSV structure...")
        pdf_text = self.read_pdf(pdf_path)
        csv_structure = self.read_csv_structure(csv_path)

        if not pdf_text or not csv_structure:
            logger.error("Failed to analyze input data")
            return False

        # Output path
        parser_path = f"custom_parsers/{bank_name}_parser.py"

        # Main loop
        current_code = ""
        last_error = ""

        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"🔄 Attempt {attempt}/{self.max_attempts}")

            if attempt == 1:
                # Generate initial code
                logger.info("💡 Generating parser code...")
                current_code = self.generate_parser_code(bank_name, pdf_text, csv_structure)
            else:
                # Fix existing code
                logger.info("🔧 Fixing parser code...")
                current_code = self.fix_parser_code(current_code, last_error)

            if not current_code:
                logger.error("Failed to generate code")
                continue

            # Save parser
            if not self.save_parser(current_code, parser_path):
                logger.error("Failed to save parser")
                continue

            # Test parser
            logger.info("🧪 Testing parser...")
            test_result = self.test_parser(parser_path, pdf_path, csv_path)

            if test_result["success"]:
                logger.info("✅ Parser test PASSED!")
                if test_result["matches_expected"]:
                    logger.info("🎉 Output matches expected data perfectly!")
                else:
                    logger.info("⚠️ Parser works but output doesn't perfectly match expected data")

                logger.info(f"Parser saved to: {parser_path}")
                return True
            else:
                last_error = test_result["error"]
                logger.warning(f"❌ Test failed: {last_error}")

                if attempt < self.max_attempts:
                    logger.info("🔄 Will attempt to fix...")
                else:
                    logger.error("💥 Max attempts reached. Parser generation failed.")

        return False


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(description="Simple AI Agent for PDF Parser Generation")
    parser.add_argument("--target", required=True, help="Bank name (e.g., icici)")
    parser.add_argument("--pdf", help="PDF path (default: data/{target}/{target}_sample.pdf)")
    parser.add_argument("--csv", help="CSV path (default: data/{target}/expected_output.csv)")
    parser.add_argument("--api-key", help="Groq API key (or set GROQ_API_KEY)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="The Groq model to use")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get API key (check CLI arg or environment)
    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Please provide Groq API key:")
        print("   --api-key YOUR_KEY")
        print("   or set GROQ_API_KEY environment variable")
        print("\n💡 Get free API key: https://console.groq.com/keys")
        sys.exit(1)

    # Set paths
    pdf_path = args.pdf or f"data/{args.target}/{args.target}_sample.pdf"
    csv_path = args.csv or f"data/{args.target}/expected_output.csv"

    print(f"🎯 Target bank: {args.target}")
    print(f"📄 PDF path: {pdf_path}")
    print(f"📊 CSV path: {csv_path}")
    print()

    # Run agent
    agent = SimplePDFParserAgent(api_key=api_key, model=args.model)
    start_time = time.time()

    success = agent.run_agent(args.target, pdf_path, csv_path)

    elapsed = time.time() - start_time
    print(f"\n⏱️ Total time: {elapsed:.2f} seconds")

    if success:
        print("🎉 SUCCESS! Parser generated successfully!")
        print(f"\n🧪 Test your parser:")
        print(f"python -c \"from custom_parsers.{args.target}_parser import parse; print(parse('{pdf_path}'))\"")
    else:
        print("💥 FAILED! Could not generate working parser.")
        print("\n🔍 Debug tips:")
        print("1. Check your PDF format")
        print("2. Verify CSV structure matches your data")
        print("3. Try with --verbose flag")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
