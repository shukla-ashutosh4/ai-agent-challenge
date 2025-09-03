# AI Agent for Bank Statement Parser Generation 🤖

An autonomous AI agent that analyzes bank statement PDFs and automatically generates custom Python parsers to extract transaction data into CSV format.

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

**Option A: Google Gemini**
```bash
# Get your API key from: https://makersuite.google.com/app/apikey
export GEMINI_API_KEY="your-api-key-here"
```

**Option B: Groq**
```bash
# Get your API key from: https://console.groq.com/keys
export GROQ_API_KEY="your-api-key-here"
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

## Adding New Banks

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
- **Production Ready**: Generated parsers include error handling and logging

## Troubleshooting

- **PDF reading errors**: Ensure `pdfplumber` is installed: `pip install pdfplumber`
- **API errors**: Check your API key is valid and has credits
- **Parser failures**: The agent will attempt 3 times; check logs for specific errors
- **Memory issues**: For large PDFs, consider using `--provider groq` for faster processing

## Demo Video

Watch the agent in action: [60-second demo showing fresh clone → agent.py → green pytest]

---

Built with ❤️ for the Agent-as-Coder Challenge
