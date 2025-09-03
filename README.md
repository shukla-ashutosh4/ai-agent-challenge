# AI Agent for Automatic Bank Statement Parser Generation

This project features a Python-based AI agent that automates the creation of parsers for bank statements. The agent uses Gemini's model to analyze sample PDF statements and generate custom Python code to extract transaction data into a structured format.

---

## 🚀 My Learning Journey & Acknowledgments

This project was my first venture into the world of AI agents and complex document parsing. It was a fantastic learning experience that pushed me to understand the practical applications of Large Language Models (LLMs) in solving real-world data extraction problems.  

I learned a great deal about **prompt engineering**, **iterative code generation**, and the challenges of creating robust, self-correcting systems.  

🙏 I would like to extend a heartfelt thank you to the entire team at **Karbon Card** for this incredible opportunity. The challenge was both stimulating and educational, providing a perfect platform to grow my skills.

---

## 📌 Project Status: Partially Successful

The agent is currently able to perform its core loop of analyzing, generating, and testing code. However, it struggles with the **self-correction phase**.

- ✅ **What Works:**  
  The agent successfully reads the PDF and CSV, sends the data to the Gemini API, and generates an initial Python parser.  

- ⚠️ **What Needs Improvement:**  
  The generated code often fails to extract the transaction data correctly, resulting in an empty DataFrame. While the agent correctly identifies this failure, its attempts to fix the code are currently unsuccessful, leading it to exhaust its attempts.  

This is a challenging problem, and the current state represents a solid foundation for future improvements in the self-correction logic.

---

## ⚙️ How It Works: The Agent's Workflow

The agent operates on a **generate → test → fix loop**.  
The core idea is to use an LLM not just to write code once, but to **iteratively refine it** based on real-world test results until it works correctly.

1. **Analysis:**  
   The agent first reads a sample PDF bank statement and a corresponding CSV file that defines the desired output structure. It sends this information to the LLaMA model to get a high-level analysis of the PDF's layout and data patterns.

2. **Code Generation:**  
   Based on this analysis, the agent prompts LLaMA to write a complete Python parser function using the `pdfplumber` library.

3. **Testing:**  
   The newly generated code is saved to a file and executed in a separate, isolated process. The script runs the parser on the sample PDF and compares its output to the expected CSV.

4. **Self-Correction:**  
   If the test fails, the agent enters the "fix" cycle. It creates a new, more detailed prompt that includes the flawed code, the exact error message from the test, and the original PDF text. This gives the AI all the context it needs to debug and provide a corrected version of the code.

5. **Loop or Succeed:**  
   The agent repeats the test-and-fix cycle up to a maximum number of attempts. If the parser passes the test, the agent saves the final code and exits successfully.

---

## Architecture Overview

The agent operates in a self-correcting loop: **Plan → Generate Code → Test → Fix → Repeat**. It analyzes PDF structure, understands the expected CSV format, generates parsing code, tests it against ground truth, and automatically fixes errors through multiple iterations until success.

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Analyze   │────▶│   Generate   │────▶│    Test     │
│     PDF     │     │    Parser    │     │   Parser    │
└─────────────┘     └──────────────┘     └─────────────┘
       ▲                                         │
       │                                         ▼
       │            ┌──────────────┐     ┌─────────────┐
       └────────────│     Fix      │◀────│   Failed?   │
                    │    Errors    │     │  (max 3x)   │
                    └──────────────┘     └─────────────┘
```

## Quick Start (5 Steps)

### 1. Clone and Navigate
```bash
git clone https://github.com/apurv-korefi/ai-agent-challenge.git
cd ai-agent-challenge
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Key
Choose either Google Gemini (recommended for free credits) or Groq:

**Google Gemini**
```bash
# Get your API key from: https://makersuite.google.com/app/apikey
export GEMINI_API_KEY="your-api-key-here"
```


### 4. Run the Agent
```bash
# For ICICI bank parser generation
python agent.py --target icici --provider gemini

# Or use Groq
python agent.py --target icici --provider groq
```

### 5. Test the Generated Parser
```bash
pytest test_parser.py -v
```

## How It Works

1. **Analysis Phase**: The agent reads the sample PDF and CSV to understand the data structure
2. **Generation Phase**: Uses LLM to generate Python code that extracts data from PDFs
3. **Testing Phase**: Runs the generated parser and compares output with expected CSV
4. **Self-Correction**: If tests fail, the agent analyzes errors and regenerates code (up to 3 attempts)
5. **Output**: Saves working parser to `custom_parsers/{bank}_parser.py`

## Project Structure
```
ai-agent-challenge/
├── agent.py                    # Main agent implementation
├── data/
│   └── icici/
│       ├── icici_sample.pdf   # Sample bank statement
│       └── icici_sample.csv   # Expected output format
├── custom_parsers/
│   └── icici_parser.py        # Generated parser (created by agent)
├── test_parser.py             # Test suite for validation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Adding New Banks(Future Works)

To add support for a new bank (e.g., SBI):

1. Create data folder: `data/sbi/`
2. Add sample files: `sbi_sample.pdf` and `sbi_sample.csv`
3. Run: `python agent.py --target sbi`
4. The agent will generate: `custom_parsers/sbi_parser.py`

## Key Features

- **Autonomous Code Generation**: Writes complete parsers without human intervention
- **Self-Debugging**: Automatically fixes errors through iterative refinement
- **Multi-Provider Support**: Works with both Google Gemini and Groq LLMs
- **Extensible**: Easy to add new banks by providing sample data

## Troubleshooting

- **PDF reading errors**: Ensure `PyPDF2` is installed: `pip install PyPDF2`
- **API errors**: Check your API key is valid and has credits
- **Parser failures**: The agent will attempt 3 times; check logs for specific errors

## Demo Video

Watch the agent in action: [https://drive.google.com/file/d/1bvK_QfK_4XMVp4iOgPbzUCHQ9LhRlO_g/view?usp=sharing]

---

Built with ❤️ for the Agent-as-Coder Challenge
By Ashutosh Shukla[https://www.linkedin.com/in/ashutosh-shukla4]
