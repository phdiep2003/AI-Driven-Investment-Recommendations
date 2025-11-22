# AI-Driven Investment Recommendation System

## Overview

The **AI-Driven Investment Recommendation System** is a Streamlit-based application that gathers a detailed customer investment profile and processes it through multiple AI agents to generate:

- Personalized portfolio recommendations  
- Quant-driven performance expectations  
- Full risk assessment  
- ESG & ethical screening results  
- Shariah compliance assessment  
- Backtesting visualizations  
- Sector allocation insights  
- Narrative suitability reports  

The project blends quantitative finance techniques with large-language-model reasoning to assist in building client-appropriate investment strategies.

---

## Prerequisites

Before running the application, ensure you have:

- **Python 3.9+**
- **pip** (Python package manager)

Optional but recommended — create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

1. Load the Open AI API key to the os environment

```bash
export OPENAI_API_KEY="your-api-key"
```
2. Install Dependencies

All required libraries are listed in requirements.txt.

Install them with:
```bash
pip install -r requirements.txt
```
This includes Streamlit, Pydantic, Plotly, and other packages used by the application.
3. Run the Streamlit App

To start the application, run:
```bash
streamlit run app.py
```
Streamlit will open the application in your default web browser.
If it does not open automatically, visit:
```bash
http://localhost:8501
```

## Project Structure
```powershell
AI-Driven-Investment-Recommendations/
│
├── app.py 
├── requirements.txt 
├── README.md 
├── sp500_static.parquet
├── cfa_standards_full.json
├── cache_fast/
```

## Key Features

✔ Client Profile form backed by a Pydantic schema

✔ AI-driven investment recommendation engine

✔ Agent-based narrative reporting

✔ Asset allocation charts (Pie, Treemap, Tables)

✔ Backtesting of portfolio strategies

✔ ESG, Shariah, ethical analysis

✔ Loadable sample profiles

✔ Session-state persistence

## License — Apache 2.0

This project is licensed under the Apache License 2.0.

You may include the full license text by creating a LICENSE file with the following header:

Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/


For more details, see the official license page:
https://www.apache.org/licenses/LICENSE-2.0

## Contact

If you have questions, suggestions, or want to collaborate, feel free to reach out.