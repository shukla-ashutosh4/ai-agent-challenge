# # #!/usr/bin/env python3
# # """
# # Simplified AI Agent for PDF Parser Generation (Groq version)

# # import argparse
# # import os
# # import sys
# # import json
# # import logging
# # from pathlib import Path
# # import time
# # import traceback

# # # PDF processing
# # import PyPDF2
# # import pandas as pd

# # # Groq AI client
# # from groq import Groq
# # import groq as groq_sdk  # for exceptions and types if needed

# # # Configure logging
# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# # logger = logging.getLogger(__name__)


# # class SimplePDFParserAgent:
# #     """
# #     Simplified version of the PDF parser agent (Groq client)
# #     Easier to understand and debug
# #     """

# #     def __init__(self, api_key: str, model: str | None = None):
# #         """Initialize the agent"""
# #         # Create Groq client. The client will look for GROQ_API_KEY by default if api_key is None.
# #         if not api_key:
# #             raise ValueError("API key is required to initialize Groq client")
# #         self.client = Groq(api_key=api_key)
# #         # Default model can be overridden by env var GROQ_MODEL or constructor argument
# #         self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
# #         self.max_attempts = 3

# #     def read_pdf(self, pdf_path: str) -> str:
# #         """Extract text from PDF"""
# #         try:
# #             with open(pdf_path, "rb") as file:
# #                 reader = PyPDF2.PdfReader(file)
# #                 text = ""
# #                 for page in reader.pages:
# #                     page_text = page.extract_text() or ""
# #                     text += page_text + "\n"
# #             logger.info(f"Extracted {len(text)} characters from PDF")
# #             return text[:2000]  # First 2000 chars for analysis
# #         except Exception as e:
# #             logger.error(f"Error reading PDF: {e}")
# #             return ""

# #     def read_csv_structure(self, csv_path: str) -> dict:
# #         """Analyze expected CSV structure"""
# #         try:
# #             df = pd.read_csv(csv_path)
# #             return {
# #                 "columns": list(df.columns),
# #                 "sample_data": df.head(3).to_dict("records"),
# #                 "shape": df.shape,
# #             }
# #         except Exception as e:
# #             logger.error(f"Error reading CSV: {e}")
# #             return {}

# #     def _call_groq(self, prompt: str, max_tokens: int = 8192) -> str:
# #         """Call Groq chat completions API and return assistant content text"""
# #         try:
# #             response = self.client.chat.completions.create(
# #                 messages=[
# #                     {"role": "system", "content": "You are an expert Python developer."},
# #                     {"role": "user", "content": prompt},
# #                 ],
# #                 model=self.model,
# #             )
# #             # Pull assistant text
# #             # The Groq python client uses response.choices[0].message.content in the README
# #             content = ""
# #             if hasattr(response, "choices") and len(response.choices) > 0:
# #                 # Some responses nest differently — try to extract defensively
# #                 choice = response.choices[0]
# #                 if hasattr(choice, "message") and getattr(choice.message, "content", None) is not None:
# #                     content = choice.message.content
# #                 elif getattr(choice, "text", None) is not None:
# #                     content = choice.text
# #                 else:
# #                     # Fallback to stringifying the response
# #                     content = str(response)
# #             else:
# #                 content = str(response)
# #             return content or ""
# #         except groq_sdk.APIError as e:
# #             logger.error(f"Groq API error: {e}")
# #             raise
# #         except Exception as e:
# #             logger.error(f"Unexpected error calling Groq: {e}")
# #             raise

# #     def generate_parser_code(self, bank_name: str, pdf_text: str, csv_structure: dict) -> str:
# #         """Generate parser code using Groq LLM"""
# #         prompt = f"""
# # You are an expert Python developer. Create a PDF parser for {bank_name} bank statements.

# # PDF SAMPLE TEXT:
# # ```
# # {pdf_text}
# # ```

# # REQUIRED OUTPUT STRUCTURE:
# # - Columns: {csv_structure.get('columns', [])}
# # - Sample data: {json.dumps(csv_structure.get('sample_data', []), indent=2)}

# # Create a complete Python function with this EXACT signature:

# # ```python
# # import PyPDF2
# # import pandas as pd
# # import re
# # from typing import List, Dict

# # def parse(pdf_path: str) -> pd.DataFrame:
# #     \"\"\"
# #     Parse {bank_name} bank statement PDF.

# #     Args:
# #         pdf_path: Path to PDF file

# #     Returns:
# #         DataFrame with columns: {csv_structure.get('columns', [])}
# #     \"\"\"
# #     # Your implementation here
# #     pass
# # ```

# # REQUIREMENTS:
# # 1. Use PyPDF2 or pdfplumber to read the PDF
# # 2. Use regex patterns or table extraction to extract data
# # 3. Return a pandas DataFrame with the exact columns specified
# # 4. Handle errors gracefully
# # 5. The function MUST be named 'parse'

# # Generate ONLY the complete working Python code. No explanations.
# # """
# #         try:
# #             logger.debug("Calling Groq to generate parser code...")
# #             code = self._call_groq(prompt)
# #             # Clean up markdown formatting if present
# #             if "```python" in code:
# #                 code = code.split("```python", 1)[1].split("```", 1)[0]
# #             elif "```" in code:
# #                 code = code.split("```", 1)[1].split("```", 1)[0]
# #             return code.strip()
# #         except Exception as e:
# #             logger.error(f"Error generating code: {e}")
# #             return ""

# #     def save_parser(self, code: str, output_path: str) -> bool:
# #         """Save generated parser code"""
# #         try:
# #             os.makedirs(os.path.dirname(output_path), exist_ok=True)
# #             with open(output_path, "w", encoding="utf-8") as f:
# #                 f.write(code)
# #             logger.info(f"Saved parser to {output_path}")
# #             return True
# #         except Exception as e:
# #             logger.error(f"Error saving parser: {e}")
# #             return False

# #     def test_parser(self, parser_path: str, pdf_path: str, csv_path: str) -> dict:
# #         """Test the generated parser"""
# #         result = {"success": False, "error": None, "matches_expected": False}

# #         try:
# #             # Import the parser
# #             sys.path.insert(0, os.path.dirname(parser_path))
# #             module_name = Path(parser_path).stem

# #             # Remove from cache if exists
# #             if module_name in sys.modules:
# #                 del sys.modules[module_name]

# #             # Import and test
# #             parser_module = __import__(module_name)

# #             # Run parser
# #             parsed_df = parser_module.parse(pdf_path)

# #             # Load expected data
# #             expected_df = pd.read_csv(csv_path)

# #             # Check results
# #             if isinstance(parsed_df, pd.DataFrame):
# #                 if list(parsed_df.columns) == list(expected_df.columns):
# #                     result["success"] = True
# #                     if parsed_df.shape == expected_df.shape:
# #                         result["matches_expected"] = True
# #                 else:
# #                     result[
# #                         "error"
# #                     ] = f"Column mismatch. Got: {list(parsed_df.columns)}, Expected: {list(expected_df.columns)}"
# #             else:
# #                 result["error"] = "Parser did not return a DataFrame"

# #         except Exception as e:
# #             result["error"] = f"Test failed: {str(e)}\n{traceback.format_exc()}"

# #         return result

# #     def fix_parser_code(self, original_code: str, error_message: str) -> str:
# #         """Generate fixed parser code using Groq"""
# #         prompt = f"""
# # The following Python parser code failed with an error:

# # ERROR: {error_message}

# # ORIGINAL CODE:
# # ```python
# # {original_code}
# # ```

# # Please fix the code to resolve this error. Return ONLY the corrected Python code without any markdown formatting.

# # Requirements:
# # 1. Keep the same function signature: def parse(pdf_path: str) -> pd.DataFrame
# # 2. Fix the specific error mentioned
# # 3. Ensure robust error handling
# # 4. Return complete working code
# # """
# #         try:
# #             logger.debug("Calling Groq to fix parser code...")
# #             fixed = self._call_groq(prompt)
# #             # Clean up markdown if present
# #             if "```python" in fixed:
# #                 fixed = fixed.split("```python", 1)[1].split("```", 1)[0]
# #             elif "```" in fixed:
# #                 fixed = fixed.split("```", 1)[1].split("```", 1)[0]
# #             return fixed.strip()
# #         except Exception as e:
# #             logger.error(f"Error fixing code: {e}")
# #             return original_code

# #     def run_agent(self, bank_name: str, pdf_path: str, csv_path: str) -> bool:
# #         """
# #         Main agent loop: Generate → Test → Fix → Repeat
# #         """

# #         logger.info(f"🚀 Starting agent for {bank_name} parser...")

# #         # Validate inputs
# #         if not os.path.exists(pdf_path):
# #             logger.error(f"PDF not found: {pdf_path}")
# #             return False

# #         if not os.path.exists(csv_path):
# #             logger.error(f"CSV not found: {csv_path}")
# #             return False

# #         # Analyze input data
# #         logger.info("📖 Analyzing PDF and CSV structure...")
# #         pdf_text = self.read_pdf(pdf_path)
# #         csv_structure = self.read_csv_structure(csv_path)

# #         if not pdf_text or not csv_structure:
# #             logger.error("Failed to analyze input data")
# #             return False

# #         # Output path
# #         parser_path = f"custom_parsers/{bank_name}_parser.py"

# #         # Main loop
# #         current_code = ""
# #         last_error = ""

# #         for attempt in range(1, self.max_attempts + 1):
# #             logger.info(f"🔄 Attempt {attempt}/{self.max_attempts}")

# #             if attempt == 1:
# #                 # Generate initial code
# #                 logger.info("💡 Generating parser code...")
# #                 current_code = self.generate_parser_code(bank_name, pdf_text, csv_structure)
# #             else:
# #                 # Fix existing code
# #                 logger.info("🔧 Fixing parser code...")
# #                 current_code = self.fix_parser_code(current_code, last_error)

# #             if not current_code:
# #                 logger.error("Failed to generate code")
# #                 continue

# #             # Save parser
# #             if not self.save_parser(current_code, parser_path):
# #                 logger.error("Failed to save parser")
# #                 continue

# #             # Test parser
# #             logger.info("🧪 Testing parser...")
# #             test_result = self.test_parser(parser_path, pdf_path, csv_path)

# #             if test_result["success"]:
# #                 logger.info("✅ Parser test PASSED!")
# #                 if test_result["matches_expected"]:
# #                     logger.info("🎉 Output matches expected data perfectly!")
# #                 else:
# #                     logger.info("⚠️ Parser works but output doesn't perfectly match expected data")

# #                 logger.info(f"Parser saved to: {parser_path}")
# #                 return True
# #             else:
# #                 last_error = test_result["error"]
# #                 logger.warning(f"❌ Test failed: {last_error}")

# #                 if attempt < self.max_attempts:
# #                     logger.info("🔄 Will attempt to fix...")
# #                 else:
# #                     logger.error("💥 Max attempts reached. Parser generation failed.")

# #         return False


# # def main():
# #     """CLI interface"""
# #     parser = argparse.ArgumentParser(description="Simple AI Agent for PDF Parser Generation")
# #     parser.add_argument("--target", required=True, help="Bank name (e.g., icici)")
# #     parser.add_argument("--pdf", help="PDF path (default: data/{target}/{target}_sample.pdf)")
# #     parser.add_argument("--csv", help="CSV path (default: data/{target}/expected_output.csv)")
# #     parser.add_argument("--api-key", help="Groq API key (or set GROQ_API_KEY)")
# #     parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
# #     parser.add_argument("--model", default="llama-3.1-8b-instant", help="The Groq model to use")

# #     args = parser.parse_args()

# #     # Set logging level
# #     if args.verbose:
# #         logging.getLogger().setLevel(logging.DEBUG)

# #     # Get API key (check CLI arg or environment)
# #     api_key = args.api_key or os.getenv("GROQ_API_KEY")
# #     if not api_key:
# #         print("❌ Please provide Groq API key:")
# #         print("   --api-key YOUR_KEY")
# #         print("   or set GROQ_API_KEY environment variable")
# #         print("\n💡 Get free API key: https://console.groq.com/keys")
# #         sys.exit(1)

# #     # Set paths
# #     pdf_path = args.pdf or f"data/{args.target}/{args.target}_sample.pdf"
# #     csv_path = args.csv or f"data/{args.target}/expected_output.csv"

# #     print(f"🎯 Target bank: {args.target}")
# #     print(f"📄 PDF path: {pdf_path}")
# #     print(f"📊 CSV path: {csv_path}")
# #     print()

# #     # Run agent
# #     agent = SimplePDFParserAgent(api_key=api_key, model=args.model)
# #     start_time = time.time()

# #     success = agent.run_agent(args.target, pdf_path, csv_path)

# #     elapsed = time.time() - start_time
# #     print(f"\n⏱️ Total time: {elapsed:.2f} seconds")

# #     if success:
# #         print("🎉 SUCCESS! Parser generated successfully!")
# #         print(f"\n🧪 Test your parser:")
# #         print(f"python -c \"from custom_parsers.{args.target}_parser import parse; print(parse('{pdf_path}'))\"")
# #     else:
# #         print("💥 FAILED! Could not generate working parser.")
# #         print("\n🔍 Debug tips:")
# #         print("1. Check your PDF format")
# #         print("2. Verify CSV structure matches your data")
# #         print("3. Try with --verbose flag")

# #     sys.exit(0 if success else 1)


# # if __name__ == "__main__":
# #     main()





# #!/usr/bin/env python3
# """
# Simplified AI Agent for PDF Parser Generation (Google AI version)
# A more straightforward version for learning and debugging
# """

# import argparse
# import os
# import sys
# import json
# import logging
# from pathlib import Path
# import time
# import traceback

# # PDF processing
# import PyPDF2
# import pandas as pd

# # Google AI client
# import google.generativeai as genai

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


# class SimplePDFParserAgent:
#     """
#     Simplified version of the PDF parser agent (Google AI client)
#     Easier to understand and debug
#     """

#     def __init__(self, api_key: str, model: str | None = None):
#         """Initialize the agent"""
#         if not api_key:
#             raise ValueError("API key is required to initialize Google AI client")
        
#         # Configure Google AI
#         genai.configure(api_key=api_key)
        
#         # Default model can be overridden by env var GOOGLE_AI_MODEL or constructor argument
#         self.model_name = model or os.getenv("GOOGLE_AI_MODEL", "gemini-1.5-flash")
#         self.model = genai.GenerativeModel(self.model_name)
#         self.max_attempts = 3

#     def read_pdf(self, pdf_path: str) -> str:
#         """Extract text from PDF"""
#         try:
#             with open(pdf_path, "rb") as file:
#                 reader = PyPDF2.PdfReader(file)
#                 text = ""
#                 for page in reader.pages:
#                     page_text = page.extract_text() or ""
#                     text += page_text + "\n"
#             logger.info(f"Extracted {len(text)} characters from PDF")
#             return text[:2000]  # First 2000 chars for analysis
#         except Exception as e:
#             logger.error(f"Error reading PDF: {e}")
#             return ""

#     def read_csv_structure(self, csv_path: str) -> dict:
#         """Analyze expected CSV structure"""
#         try:
#             df = pd.read_csv(csv_path)
#             return {
#                 "columns": list(df.columns),
#                 "sample_data": df.head(3).to_dict("records"),
#                 "shape": df.shape,
#             }
#         except Exception as e:
#             logger.error(f"Error reading CSV: {e}")
#             return {}

#     def _call_google_ai(self, prompt: str) -> str:
#         """Call Google AI API and return response text"""
#         try:
#             # Generate response using Google AI
#             response = self.model.generate_content(
#                 prompt,
#                 generation_config=genai.types.GenerationConfig(
#                     temperature=0.1,  # Lower temperature for more consistent code
#                     max_output_tokens=8192,
#                     top_p=0.8,
#                     top_k=40
#                 )
#             )
            
#             # Extract text from response
#             if response and response.text:
#                 return response.text
#             else:
#                 logger.warning("Empty response from Google AI")
#                 return ""
                
#         except Exception as e:
#             logger.error(f"Google AI API error: {e}")
#             raise

#     def generate_parser_code(self, bank_name: str, pdf_text: str, csv_structure: dict) -> str:
#         """Generate parser code using Google AI"""
#         prompt = f"""
# You are an expert Python developer. Create a PDF parser for {bank_name} bank statements.

# PDF SAMPLE TEXT:
# ```
# {pdf_text}
# ```

# REQUIRED OUTPUT STRUCTURE:
# - Columns: {csv_structure.get('columns', [])}
# - Sample data: {json.dumps(csv_structure.get('sample_data', []), indent=2)}

# Create a complete Python function with this EXACT signature:

# ```python
# import PyPDF2
# import pandas as pd
# import re
# from typing import List, Dict

# def parse(pdf_path: str) -> pd.DataFrame:
#     \"\"\"
#     Parse {bank_name} bank statement PDF.

#     Args:
#         pdf_path: Path to PDF file

#     Returns:
#         DataFrame with columns: {csv_structure.get('columns', [])}
#     \"\"\"
#     # Your implementation here
#     pass
# ```

# REQUIREMENTS:
# 1. Use PyPDF2 or pdfplumber to read the PDF
# 2. Use regex patterns or table extraction to extract data
# 3. Return a pandas DataFrame with the exact columns specified
# 4. Handle errors gracefully
# 5. The function MUST be named 'parse'
# 6. IMPORTANT: Use 'import pandas as pd', not 'import pd'
# 7. IMPORTANT: If using pdfplumber, use 'pdfplumber.open()' not 'pandas.open()' or 'pd.open()'

# Generate ONLY the complete working Python code. No explanations or markdown formatting.
# """
#         try:
#             logger.debug("Calling Google AI to generate parser code...")
#             code = self._call_google_ai(prompt)
            
#             # Clean up markdown formatting if present
#             if "```python" in code:
#                 code = code.split("```python", 1)[1].split("```", 1)[0]
#             elif "```" in code:
#                 code = code.split("```", 1)[1].split("```", 1)[0]
            
#             # Apply basic fixes
#             code = self._apply_basic_fixes(code)
            
#             return code.strip()
#         except Exception as e:
#             logger.error(f"Error generating code: {e}")
#             return ""

#     def _apply_basic_fixes(self, code: str) -> str:
#         """Apply basic fixes to generated code"""
#         # Fix common import issues
#         code = code.replace("import pd", "import pandas as pd")
        
#         # Fix pdfplumber usage issues
#         code = code.replace("pandas.open(", "pdfplumber.open(")
#         code = code.replace("pd.open(", "pdfplumber.open(")
        
#         # Ensure pdfplumber import if needed
#         if "pdfplumber.open(" in code and "import pdfplumber" not in code:
#             code = code.replace("import pandas as pd", "import pandas as pd\nimport pdfplumber")
        
#         # Remove outdated PyPDF2 references
#         code = code.replace("PyPDF2.PdfReadError", "Exception")
        
#         return code

#     def save_parser(self, code: str, output_path: str) -> bool:
#         """Save generated parser code"""
#         try:
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             with open(output_path, "w", encoding="utf-8") as f:
#                 f.write(code)
#             logger.info(f"Saved parser to {output_path}")
#             return True
#         except Exception as e:
#             logger.error(f"Error saving parser: {e}")
#             return False

#     def test_parser(self, parser_path: str, pdf_path: str, csv_path: str) -> dict:
#         """Test the generated parser"""
#         result = {"success": False, "error": None, "matches_expected": False}

#         try:
#             # Import the parser
#             sys.path.insert(0, os.path.dirname(parser_path))
#             module_name = Path(parser_path).stem

#             # Remove from cache if exists
#             if module_name in sys.modules:
#                 del sys.modules[module_name]

#             # Import and test
#             parser_module = __import__(module_name)

#             # Run parser
#             parsed_df = parser_module.parse(pdf_path)

#             # Load expected data
#             expected_df = pd.read_csv(csv_path)

#             # Check results
#             if isinstance(parsed_df, pd.DataFrame):
#                 if list(parsed_df.columns) == list(expected_df.columns):
#                     result["success"] = True
#                     if parsed_df.shape == expected_df.shape:
#                         result["matches_expected"] = True
#                 else:
#                     result[
#                         "error"
#                     ] = f"Column mismatch. Got: {list(parsed_df.columns)}, Expected: {list(expected_df.columns)}"
#             else:
#                 result["error"] = "Parser did not return a DataFrame"

#         except Exception as e:
#             result["error"] = f"Test failed: {str(e)}\n{traceback.format_exc()}"

#         return result

#     def fix_parser_code(self, original_code: str, error_message: str) -> str:
#         """Generate fixed parser code using Google AI"""
#         prompt = f"""
# The following Python parser code failed with an error:

# ERROR: {error_message}

# ORIGINAL CODE:
# ```python
# {original_code}
# ```

# Please fix the code to resolve this error. Return ONLY the corrected Python code without any markdown formatting.

# Requirements:
# 1. Keep the same function signature: def parse(pdf_path: str) -> pd.DataFrame
# 2. Fix the specific error mentioned
# 3. Ensure robust error handling
# 4. Return complete working code
# 5. IMPORTANT: Use 'import pandas as pd', not 'import pd'
# 6. IMPORTANT: If using pdfplumber, use 'pdfplumber.open()' not 'pandas.open()' or 'pd.open()'

# Common fixes needed:
# - Use 'pdfplumber.open()' instead of 'pandas.open()' or 'pd.open()'
# - Import pdfplumber if using pdfplumber functions
# - Use 'Exception' instead of 'PyPDF2.PdfReadError'
# - Ensure all required columns are present in output DataFrame
# """
#         try:
#             logger.debug("Calling Google AI to fix parser code...")
#             fixed = self._call_google_ai(prompt)
            
#             # Clean up markdown if present
#             if "```python" in fixed:
#                 fixed = fixed.split("```python", 1)[1].split("```", 1)[0]
#             elif "```" in fixed:
#                 fixed = fixed.split("```", 1)[1].split("```", 1)[0]
            
#             # Apply basic fixes
#             fixed = self._apply_basic_fixes(fixed)
            
#             return fixed.strip()
#         except Exception as e:
#             logger.error(f"Error fixing code: {e}")
#             return original_code

#     def run_agent(self, bank_name: str, pdf_path: str, csv_path: str) -> bool:
#         """
#         Main agent loop: Generate → Test → Fix → Repeat
#         """

#         logger.info(f"🚀 Starting agent for {bank_name} parser...")

#         # Validate inputs
#         if not os.path.exists(pdf_path):
#             logger.error(f"PDF not found: {pdf_path}")
#             return False

#         if not os.path.exists(csv_path):
#             logger.error(f"CSV not found: {csv_path}")
#             return False

#         # Analyze input data
#         logger.info("📖 Analyzing PDF and CSV structure...")
#         pdf_text = self.read_pdf(pdf_path)
#         csv_structure = self.read_csv_structure(csv_path)

#         if not pdf_text or not csv_structure:
#             logger.error("Failed to analyze input data")
#             return False

#         # Output path
#         parser_path = f"custom_parsers/{bank_name}_parser.py"
#         os.makedirs("custom_parsers", exist_ok=True)

#         # Main loop
#         current_code = ""
#         last_error = ""

#         for attempt in range(1, self.max_attempts + 1):
#             logger.info(f"🔄 Attempt {attempt}/{self.max_attempts}")

#             if attempt == 1:
#                 # Generate initial code
#                 logger.info("💡 Generating parser code...")
#                 current_code = self.generate_parser_code(bank_name, pdf_text, csv_structure)
#             else:
#                 # Fix existing code
#                 logger.info("🔧 Fixing parser code...")
#                 current_code = self.fix_parser_code(current_code, last_error)

#             if not current_code:
#                 logger.error("Failed to generate code")
#                 continue

#             # Save parser
#             if not self.save_parser(current_code, parser_path):
#                 logger.error("Failed to save parser")
#                 continue

#             # Test parser
#             logger.info("🧪 Testing parser...")
#             test_result = self.test_parser(parser_path, pdf_path, csv_path)

#             if test_result["success"]:
#                 logger.info("✅ Parser test PASSED!")
#                 if test_result["matches_expected"]:
#                     logger.info("🎉 Output matches expected data perfectly!")
#                 else:
#                     logger.info("⚠️ Parser works but output doesn't perfectly match expected data")

#                 logger.info(f"Parser saved to: {parser_path}")
#                 return True
#             else:
#                 last_error = test_result["error"]
#                 logger.warning(f"❌ Test failed: {last_error}")

#                 if attempt < self.max_attempts:
#                     logger.info("🔄 Will attempt to fix...")
#                     time.sleep(1)  # Brief pause between attempts
#                 else:
#                     logger.error("💥 Max attempts reached. Parser generation failed.")

#         return False


# def main():
#     """CLI interface"""
#     parser = argparse.ArgumentParser(description="Simple AI Agent for PDF Parser Generation (Google AI)")
#     parser.add_argument("--target", required=True, help="Bank name (e.g., icici)")
#     parser.add_argument("--pdf", help="PDF path (default: data/{target}/{target}_sample.pdf)")
#     parser.add_argument("--csv", help="CSV path (default: data/{target}/expected_output.csv)")
#     parser.add_argument("--api-key", help="Google AI API key (or set GOOGLE_AI_API_KEY)")
#     parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
#     parser.add_argument("--model", default="gemini-1.5-flash", help="The Google AI model to use")

#     args = parser.parse_args()

#     # Set logging level
#     if args.verbose:
#         logging.getLogger().setLevel(logging.DEBUG)

#     # Get API key (check CLI arg or environment)
#     api_key = args.api_key or os.getenv("GOOGLE_AI_API_KEY")
#     if not api_key:
#         print("❌ Please provide Google AI API key:")
#         print("   --api-key YOUR_KEY")
#         print("   or set GOOGLE_AI_API_KEY environment variable")
#         print("\n💡 Get free API key: https://aistudio.google.com/app/apikey")
#         sys.exit(1)

#     # Set paths
#     pdf_path = args.pdf or f"data/{args.target}/{args.target}_sample.pdf"
#     csv_path = args.csv or f"data/{args.target}/expected_output.csv"

#     print(f"🎯 Target bank: {args.target}")
#     print(f"📄 PDF path: {pdf_path}")
#     print(f"📊 CSV path: {csv_path}")
#     print(f"🤖 Model: {args.model}")
#     print()

#     # Run agent
#     try:
#         agent = SimplePDFParserAgent(api_key=api_key, model=args.model)
#         start_time = time.time()

#         success = agent.run_agent(args.target, pdf_path, csv_path)

#         elapsed = time.time() - start_time
#         print(f"\n⏱️ Total time: {elapsed:.2f} seconds")

#         if success:
#             print("🎉 SUCCESS! Parser generated successfully!")
#             print(f"\n🧪 Test your parser:")
#             print(f"python -c \"from custom_parsers.{args.target}_parser import parse; print(parse('{pdf_path}'))\"")
#         else:
#             print("💥 FAILED! Could not generate working parser.")
#             print("\n🔍 Debug tips:")
#             print("1. Check your PDF format")
#             print("2. Verify CSV structure matches your data")
#             print("3. Try with --verbose flag")

#         sys.exit(0 if success else 1)
        
#     except Exception as e:
#         print(f"❌ Error initializing agent: {e}")
#         print("\n🔍 Common issues:")
#         print("1. Invalid API key - get one from https://aistudio.google.com/app/apikey")
#         print("2. Network connectivity issues")
#         print("3. Missing google-generativeai package - install with: pip install google-generativeai")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
"""
AI Agent for Automatic Bank Statement Parser Generation (Gemini-only version)
This agent analyzes bank statement PDFs and their corresponding CSVs,
then generates custom Python parsers that can extract and format the data correctly.
"""

import os
import sys
import json
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass, field
import re
import time

# PDF and AI dependencies
import pdfplumber
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Maintains the current state of the agent's execution"""
    target_bank: str
    pdf_path: Path
    csv_path: Path
    parser_path: Path
    attempts: int = 0
    max_attempts: int = 3
    conversation_history: List[Dict] = field(default_factory=list)
    generated_code: str = ""
    test_results: Dict = field(default_factory=dict)


class ParserGeneratorAgent:
    """Main agent that generates bank statement parsers using Gemini"""

    def __init__(self, api_key: str):
        """Initializes the agent and the Gemini model."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def _call_gemini(self, prompt: str) -> str:
        """Calls the Gemini API and returns the generated text."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return ""

    def analyze_pdf_structure(self, state: AgentState) -> str:
        """Analyzes the PDF and CSV to understand the required parsing logic."""
        logger.info("📊 Analyzing PDF and CSV structure...")
        
        pdf_content = self._read_pdf_sample(state.pdf_path)
        csv_df = pd.read_csv(state.csv_path)
        csv_info = {
            "columns": list(csv_df.columns),
            "dtypes": {col: str(dtype) for col, dtype in csv_df.dtypes.items()},
            "sample_rows": csv_df.head(3).to_dict('records'),
            "shape": csv_df.shape
        }
        
        prompt = f"""
        I need to create a Python parser for a bank statement PDF.

        PDF Sample Content (first 2 pages):
        ```
        {pdf_content}
        ```
        
        Expected CSV Output Format:
        - Columns: {csv_info['columns']}
        - Data Types: {csv_info['dtypes']}
        - Sample Rows: {json.dumps(csv_info['sample_rows'], indent=2)}
        
        Please provide a detailed analysis of the PDF's structure. Focus on:
        1.  The regex pattern needed to capture a single transaction row.
        2.  Date, debit, credit, and description formats.
        3.  Any headers, footers, or metadata that should be ignored.
        """
        
        analysis = self._call_gemini(prompt)
        state.conversation_history.append({"role": "analysis", "content": analysis})
        return analysis

    def generate_parser_code(self, state: AgentState, analysis: str) -> str:
        """Generates the parser code based on the initial analysis."""
        logger.info("🔨 Generating initial parser code...")
        
        csv_df = pd.read_csv(state.csv_path)
        
        prompt = f"""
        Generate a complete Python parser for {state.target_bank.upper()} bank statements based on the following analysis.

        Analysis:
        {analysis}
        
        The parser MUST adhere to these rules:
        1.  It must contain a function with the exact signature: `def parse(pdf_path: str) -> pd.DataFrame:`.
        2.  It must return a pandas DataFrame with these EXACT columns: {list(csv_df.columns)}.
        3.  It must use the `pdfplumber` library to read the PDF.
        4.  CRITICAL: If no transactions are found or an error occurs, you MUST return an empty DataFrame with the correct columns. Example: `return pd.DataFrame(columns={list(csv_df.columns)})`.
        
        Generate ONLY the complete, runnable Python code inside a markdown block. No explanations.
        
        Template:
        ```python
        import pdfplumber
        import pandas as pd
        import re
        from datetime import datetime
        import logging

        logger = logging.getLogger(__name__)

        def parse(pdf_path: str) -> pd.DataFrame:
            \"\"\"Parse {state.target_bank.upper()} bank statement PDF.\"\"\"
            # Your full implementation here
            pass
        ```
        """
        
        code = self._call_gemini(prompt)
        code = self._extract_python_code(code)
        
        state.generated_code = code
        state.conversation_history.append({"role": "code_generation", "content": code})
        
        return code

    def test_parser(self, state: AgentState) -> Tuple[bool, str]:
        """Tests the generated parser in a separate process."""
        logger.info("🧪 Testing generated parser...")

        state.parser_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state.parser_path, 'w', encoding='utf-8') as f:
            f.write(state.generated_code)

        test_script_path = Path("test_generated_parser.py")
        with open(test_script_path, 'w', encoding='utf-8') as f:
            f.write(f"""
import sys
import pandas as pd
from pathlib import Path

try:
    sys.path.insert(0, r'{state.parser_path.parent.resolve()}')
    from {state.parser_path.stem} import parse
    
    result_df = parse(r'{state.pdf_path.resolve()}')
    expected_df = pd.read_csv(r'{state.csv_path.resolve()}')
    
    if not result_df.equals(expected_df):
        print("FAILURE: DataFrames do not match.")
        print(f"Result columns: {{result_df.columns.tolist()}}")
        print(f"Expected columns: {{expected_df.columns.tolist()}}")
        print(f"Result shape: {{result_df.shape}} vs Expected: {{expected_df.shape}}")
        sys.exit(1)
        
    print("SUCCESS: DataFrames match exactly!")
    sys.exit(0)
    
except Exception as e:
    import traceback
    print(f"ERROR: An exception occurred during testing.")
    traceback.print_exc()
    sys.exit(2)
""")

        try:
            result = subprocess.run(
                [sys.executable, str(test_script_path)],
                capture_output=True, text=True, timeout=30
            )
            success = result.returncode == 0
            output = result.stdout + result.stderr
            state.test_results = {"success": success, "output": output}
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Test timed out after 30 seconds."
        finally:
            test_script_path.unlink(missing_ok=True)

    def fix_parser_errors(self, state: AgentState, error_output: str, pdf_text: str) -> str:
        """Attempts to fix the parser code based on test failure output."""
        logger.info("🔧 Attempting to fix parser errors...")
        
        csv_df = pd.read_csv(state.csv_path)

        prompt = f"""
        The generated Python parser failed the test. You must debug and fix it.

        Current flawed code:
        ```python
        {state.generated_code}
        ```
        
        Here is the error output from the test:
        ```
        {error_output}
        ```
        
        The error shows the parser produced an empty DataFrame (`Result shape: (0, 0)`) or one with no columns (`Result columns: []`). This means the regex failed to extract data AND the error handling is wrong.
        You MUST re-analyze the original PDF text to create a correct regex pattern.

        Original PDF Text Sample:
        ```
        {pdf_text}
        ```

        Required Columns: {list(csv_df.columns)}

        Instructions for the fix:
        1.  Carefully examine the `Original PDF Text Sample` to find the correct pattern for transaction lines.
        2.  Write a new, robust regular expression to capture the data.
        3.  Ensure the final DataFrame has columns that EXACTLY match the required columns.
        4.  **CRITICAL FIX**: If no transactions are found, or in any `except` block, the function MUST return a correctly initialized empty DataFrame: `pd.DataFrame(columns={list(csv_df.columns)})`.
        
        Generate the COMPLETE FIXED Python code. Return ONLY the code, with no explanations.
        """
        
        fixed_code = self._call_gemini(prompt)
        fixed_code = self._extract_python_code(fixed_code)
        
        state.generated_code = fixed_code
        state.conversation_history.append({"role": "fix_attempt", "content": fixed_code})
        
        return fixed_code

    def run(self, state: AgentState) -> bool:
        """Main agent execution loop: Analyze -> Generate -> Test -> Fix."""
        logger.info(f"🚀 Starting agent for {state.target_bank} bank parser generation")
        
        pdf_text = self._read_pdf_sample(state.pdf_path)
        if not pdf_text:
            logger.error("Could not read PDF content. Aborting.")
            return False

        analysis = self.analyze_pdf_structure(state)
        self.generate_parser_code(state, analysis)
        
        while state.attempts < state.max_attempts:
            state.attempts += 1
            logger.info(f"📝 Attempt {state.attempts}/{state.max_attempts}")
            
            success, output = self.test_parser(state)
            
            if success:
                logger.info("✅ Parser generated successfully!")
                self._save_final_parser(state)
                return True
            
            logger.warning(f"Test failed. Error:\n{output}")
            if state.attempts < state.max_attempts:
                self.fix_parser_errors(state, output, pdf_text)
            else:
                logger.error(f"✅ Parser generated successfully!")
        
        return False

    def _read_pdf_sample(self, pdf_path: Path) -> str:
        """Reads the text from the first two pages of a PDF."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages[:2]:
                    text += page.extract_text() or ""
                return text
        except Exception as e:
            logger.error(f"Error reading PDF with pdfplumber: {e}")
            return ""

    def _extract_python_code(self, text: str) -> str:
        """Extracts Python code from a markdown block in the LLM's response."""
        match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        logger.warning("Could not find a python markdown block in LLM response. Returning raw text.")
        return text.strip()

    def _save_final_parser(self, state: AgentState):
        """Saves the final, successful parser code with a header."""
        logger.info(f"💾 Saving final parser to {state.parser_path}")
        header = f'''"""
Auto-generated parser for {state.target_bank.upper()} bank statements.
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
This parser was automatically created by an AI agent.
"""

'''
        with open(state.parser_path, 'w', encoding='utf-8') as f:
            f.write(header + state.generated_code)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='AI Agent for Bank Statement Parser Generation')
    parser.add_argument('--target', required=True, help='Target bank (e.g., icici)')
    parser.add_argument('--api-key', help='Gemini API key (or set GEMINI_API_KEY env var)')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        logger.error("Please provide a Gemini API key via --api-key or the GEMINI_API_KEY environment variable.")
        sys.exit(1)
        
    target_bank = args.target.lower()
    base_path = Path.cwd()
    
    state = AgentState(
        target_bank=target_bank,
        pdf_path=base_path / 'data' / target_bank / f'{target_bank}_sample.pdf',
        csv_path=base_path / 'data' / target_bank / f'expected_output.csv',
        parser_path=base_path / 'custom_parsers' / f'{target_bank}_parser.py'
    )
    
    if not state.pdf_path.exists() or not state.csv_path.exists():
        logger.error(f"Missing required files. Ensure PDF and CSV exist at {state.pdf_path.parent}")
        sys.exit(1)
        
    agent = ParserGeneratorAgent(api_key)
    success = agent.run(state)
    
    if success:
        logger.info(f"✨ Success! Parser saved to {state.parser_path}")
    else:
        logger.error("Failed to generate a working parser.")
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

