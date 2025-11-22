# ============================================================================
#  AI Portfolio Recommender (Upgraded for Full Institutional Questionnaire)
# ============================================================================

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pydantic import BaseModel
from scipy.optimize import minimize
from openai import OpenAI
import plotly.express as px

# ============================================================================
#  STREAMLIT CONFIG + DARK CHATGPT-STYLE UI
# ============================================================================

st.set_page_config(
    page_title="AI Portfolio Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp {
    background-color: #111111;
    color: #eaeaea;
}

/* General text */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff;
}
p {
    font-size: 1.1rem !important;
}
input {
    font-size: 1.1rem !important;
}

/* Chat-style bubbles */
.chat-bubble {
    background-color: #1f1f1f;
    padding: 1rem 1.2rem;
    border-radius: 15px;
    margin-bottom: 1.0rem;
    border: 1px solid #3f3f3f;
    font-size: 1.15rem !important;   /* Adjust size here */
    line-height: 1.55rem !important; /* Optional: improve readability */
}

/* Section headers */
.section-header {
    color: #10a37f;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 1.3rem;
    margin-bottom: 0.6rem;
}

/* KPI cards */
.kpi-card {
    background-color: #1b1b1b;
    padding: 1.2rem;
    border-radius: 15px;
    border: 1px solid #2f2f2f;
    text-align: center;
}
.kpi-label {
    color: #bbbbbb;
    font-size: 0.9rem;
}
.kpi-value {
    color: #ffffff;
    font-size: 1.6rem;
    font-weight: 700;
}

/* Client summary card */
.client-card {
    background: #1b1b1b;
    border-radius: 15px;
    border: 1px solid #333333;
    padding: 1rem 1.2rem;
    font-size: 1.15rem !important;   /* Adjust size here */
    line-height: 1.55rem !important; /* Optional: improve readability */
}

/* Buttons */
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 999px;
    border: none;
    padding: 0.4rem 1.2rem;
    font-weight: 600;
}

/* Inputs on dark background */
div.stTextInput>div>input,
div.stNumberInput>div>input,
div.stSelectbox>div>div>div>div,
div.stSlider>div>div>div>input {
    background-color: #222222 !important;
    color: #eaeaea !important;
}
.stSpinner > div > div {
    color: #10a37f !important;
    font-weight: 600 !important;
}

/* Spinner rotating circle */
.stSpinner > div > div::after {
    border-top-color: #10a37f !important;
    border-right-color: rgba(16,163,127,0.4) !important;
    border-bottom-color: #10a37f !important;
    border-left-color: rgba(16,163,127,0.4) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
#  MODELS
# ============================================================================


class CustomerProfile(BaseModel):
    # Client details
    full_name: str
    email: str

    # Return objectives
    primary_goal: str  # Capital preservation, Income, Long-term growth
    target_annual_return: float
    tradeoff_loss_vs_return: int  # 0–10
    tolerance_underperformance: int  # 0–10

    # Liquidity & time horizon
    withdrawal_start: str  # options string
    investment_horizon: str  # "< 3 years", "3–10 years", "> 10 years"
    desired_liquid_portion_pct: int

    # Legal & regulatory constraints
    legal_constraints: List[str]
    other_legal_notes: str

    # Tax considerations
    tax_status: str
    tax_priority: str
    tax_deferral_preference: int  # 0–10
    tax_jurisdictions_to_avoid: str

    # Ethical / ESG
    esg_exclusions: List[str]
    esg_flags: List[str]
    esg_importance: int  # 0–10
    esg_return_tradeoff: int  # 0–10

    # Diversification & allocation
    allocation_equities_pct: float
    allocation_fixed_income_pct: float
    allocation_alternatives_pct: float
    concentration_tolerance: str  # Low / Moderate / High
    max_single_issuer_exposure_pct: float
    management_style: str  # Passive / Active / Hybrid

    # Performance & reporting
    preferred_benchmark: str
    reporting_frequency: str

    # Derived for quant filters
    risk_tolerance: str  # "low" / "medium" / "high"
    islamic_investing: bool  # derived from Shariah-related flags


class Agent1Output(BaseModel):
    selected_stock_tickers: List[str]
    reasoning: str


class QuantEngineOutput(BaseModel):
    portfolio_weights: Dict[str, float]
    expected_annual_return: float
    annual_volatility: float
    optimization_details: str


class Agent2Output(BaseModel):
    suitability_explanation: str
    ethical_considerations: str
    shariah_compliance_statement: str
    risk_assessment: str
    limitations: str


# ============================================================================
# 1 — FAST S&P 500 LOADER
# ============================================================================
def load_sp500_list() -> pd.DataFrame:
    """
    Ultra-fast loader.
    Tries sp500_static.parquet first; falls back to a small static universe.
    """
    file_path = "sp500_static.parquet"

    if os.path.exists(file_path):
        try:
            df = pd.read_parquet(file_path)
            # ensure required columns are present
            for col in [
                "ticker",
                "name",
                "industry",
                "shariah_compliant",
                "roe",
                "revenue_growth",
                "beta",
            ]:
                if col not in df.columns:
                    raise ValueError("Missing required column")
            return df
        except Exception:
            pass

    # Minimal static fallback universe
    data = [
        {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "industry": "Technology",
            "shariah_compliant": True,
            "roe": 0.30,
            "revenue_growth": 0.10,
            "beta": 1.2,
        },
        {
            "ticker": "MSFT",
            "name": "Microsoft Corp.",
            "industry": "Technology",
            "shariah_compliant": True,
            "roe": 0.35,
            "revenue_growth": 0.12,
            "beta": 1.0,
        },
        {
            "ticker": "GOOGL",
            "name": "Alphabet Inc.",
            "industry": "Communication",
            "shariah_compliant": True,
            "roe": 0.18,
            "revenue_growth": 0.10,
            "beta": 1.0,
        },
        {
            "ticker": "JNJ",
            "name": "Johnson & Johnson",
            "industry": "Health Care",
            "shariah_compliant": False,
            "roe": 0.25,
            "revenue_growth": 0.05,
            "beta": 0.8,
        },
        {
            "ticker": "PG",
            "name": "Procter & Gamble",
            "industry": "Consumer Staples",
            "shariah_compliant": False,
            "roe": 0.20,
            "revenue_growth": 0.04,
            "beta": 0.7,
        },
        {
            "ticker": "TSLA",
            "name": "Tesla Inc.",
            "industry": "Consumer Discretionary",
            "shariah_compliant": True,
            "roe": 0.15,
            "revenue_growth": 0.25,
            "beta": 1.6,
        },
        {
            "ticker": "NVDA",
            "name": "NVIDIA Corp.",
            "industry": "Technology",
            "shariah_compliant": True,
            "roe": 0.40,
            "revenue_growth": 0.30,
            "beta": 1.5,
        },
        {
            "ticker": "XOM",
            "name": "Exxon Mobil",
            "industry": "Energy",
            "shariah_compliant": False,
            "roe": 0.20,
            "revenue_growth": 0.03,
            "beta": 1.1,
        },
    ]
    return pd.DataFrame(data)


# ============================================================================
# 2 — Python Pre-filter (Agent 1 helper)
# ============================================================================


def python_pre_filter_stocks(profile: CustomerProfile, df: pd.DataFrame):
    """
    Optimized:
    - Fast dataframe ops
    - Reduced columns
    - Return only TOP 25 candidates for speed
    """
    data = df.copy()

    if profile.islamic_investing:
        data = data[data["shariah_compliant"] == True]

    required = ["roe", "revenue_growth", "beta"]
    data = data.dropna(subset=required)

    if profile.risk_tolerance == "low":
        data = data[data["beta"] < 1]
        data = data[data["roe"] > 0.05]
    elif profile.risk_tolerance == "medium":
        data = data[data["beta"] < 1.3]
        data = data[data["roe"] > 0.08]
    elif profile.risk_tolerance == "high":
        data = data[data["roe"] > 0.12]
        data = data[data["revenue_growth"] > 0]

    def score(row):
        return (
            row["roe"] * 0.35
            + row["revenue_growth"] * 0.25
            + (1 / (1 + abs(row["beta"]))) * 0.15
        )

    data["quality_score"] = data.apply(score, axis=1)
    top = data.sort_values("quality_score", ascending=False).head(25)

    return top.to_dict(orient="records")


# ============================================================================
# 3 — Agent 1 (GPT) — stock selection
# ============================================================================


def agent1_gpt4o(profile: CustomerProfile, sp500: pd.DataFrame) -> Agent1Output:
    candidates = python_pre_filter_stocks(profile, sp500)
    candidates_json = json.dumps(candidates, indent=2)
    profile_json = profile.model_dump_json(indent=2)

    # Conditional text: include Shariah clause ONLY if Islamic investing is True
    shariah_clause = (
        "- Respect Shariah / Islamic requirements if indicated.\n"
        if profile.islamic_investing
        else ""
    )

    prompt = f"""
You are Agent 1, an equity portfolio construction assistant for institutional investors.

You are given:
1) A filtered set of S&P 500 stocks with basic quality metrics.
2) A full institutional investor questionnaire response in JSON.

Your task:
- Select **15–20 tickers** from the candidates.
- Make sure the selections reflect ALL aspects of the profile:
  • Full name / email only for labeling (no doxxing).
  • Return objectives (target return, risk/return trade-off, tolerance for underperformance).
  • Liquidity & time horizon.
  • Legal & regulatory constraints (ERISA, UCITS, jurisdiction limits). 
  {shariah_clause.strip()}
  • Tax status and tax preferences (capital-gains vs income, deferral preference).
  • ESG / ethical exclusions and flags.
  • Diversification and concentration tolerance (max issuer/sector).
  • Asset allocation preferences and management style.
  • Preferred benchmark and reporting cadence.
  • Any other information contained in the JSON profile.

Important:
- Respect explicit exclusions (industries, structures, jurisdictions).
- Avoid concentration that conflicts with the stated tolerance.
{shariah_clause}

Input — Filtered candidates (list of dicts):
{candidates_json}

Input — Customer profile JSON:
{profile_json}

You must respond with **ONLY** a JSON object with this exact schema:

{{
  "selected_stock_tickers": ["TICKER1", "TICKER2", "..."],
  "reasoning": "A detailed but concise explanation (CFA-level) describing why these stocks fit the profile."
}}
"""

    client = OpenAI()

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Return ONLY JSON following the schema."},
        ],
    )

    return Agent1Output.model_validate_json(res.choices[0].message.content)


# ============================================================================
# 4 — Historical price cache helper
# ============================================================================


def download_prices(tickers: List[str]) -> pd.DataFrame:
    cache_dir = "cache_fast"
    os.makedirs(cache_dir, exist_ok=True)

    today = datetime.now()
    max_age = timedelta(days=56)  # 8 weeks

    frames = []

    for t in tickers:
        fpath = os.path.join(cache_dir, f"{t}.parquet")
        df = None

        if os.path.exists(fpath):
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
            if today - mtime <= max_age:
                try:
                    df = pd.read_parquet(fpath)
                except Exception:
                    df = None

        if df is None:
            try:
                d = yf.download(
                    t,
                    period="5y",
                    interval="1mo",
                    progress=False,
                    auto_adjust=False,
                    multi_level_index=False,
                )
                if not d.empty:
                    d = d[["Adj Close"]].rename(columns={"Adj Close": t})
                    d.to_parquet(fpath)
                    df = d
            except Exception:
                continue

        if df is not None:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1).sort_index()
    return out


# ============================================================================
# 5 — Quant Engine
# ============================================================================

def quant_engine(tickers: List[str]) -> QuantEngineOutput:
    prices = download_prices(tickers)
    if prices.empty or len(prices.columns) < 2:
        return QuantEngineOutput({}, 0.0, 0.0, "Not enough historical data.")

    ret = prices.pct_change().dropna()
    mu = ret.mean() * 12
    cov = ret.cov() * 12

    n = len(mu)
    w0 = np.ones(n) / n

    def vol(w):
        return float(np.sqrt(np.dot(w.T, np.dot(cov, w))))

    bounds = [(0.01, 0.20)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    try:
        res = minimize(vol, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        w = res.x if res.success else w0
    except Exception:
        w = w0

    w = np.array(w)
    w = w / w.sum()
    er = float(np.dot(mu, w))
    stdev = vol(w)

    return QuantEngineOutput(
        portfolio_weights={t: float(w[i]) for i, t in enumerate(mu.index)},
        expected_annual_return=er,
        annual_volatility=stdev,
        optimization_details="Optimized with 5-year monthly historical data (8-week cached).",
    )


# ============================================================================
# 6 — Agent 2 (GPT) — narrative report
# ============================================================================

def agent2_gpt(
    profile: CustomerProfile,
    a1: Agent1Output,
    qout: QuantEngineOutput,
    cfa_ethical_guidelines_str: str
) -> Agent2Output:
    client = OpenAI()

    # --- PROMPT ---
    user_prompt = f"""
You are Agent 2, a CFA charterholder and institutional portfolio consultant.

Your tasks:

1. **Suitability**
   Explain how the recommended portfolio aligns with the client’s full questionnaire, including:
   - Risk tolerance
   - Investment horizon
   - Return expectations
   - Constraints (ethical, religious, liquidity)
   - Incorporate Agent 1's reasoning as needed.

2. **Ethical / ESG / Religious**
   Compare the recommendation to CFA Institute ethical guidelines.
   Include ESG factors and any religious / Shariah considerations.

3. **Shariah Compliance Statement**
   - If the client requires Shariah constraints, explicitly confirm or deny compliance.
   - If Shariah is not required, explicitly state that this is not applicable.

4. **Risk Assessment**
   Use the provided quantitative metrics:
   - Expected annual return: {qout.expected_annual_return:.4f}
   - Annual volatility: {qout.annual_volatility:.4f}
   Discuss volatility, concentration, and drawdown considerations.

5. **Limitations**
   Explain model limitations: historical data, optimization assumptions, mock data, simplifications.

---

### Customer Profile (JSON):
{profile.model_dump_json(indent=2)}

### Agent 1 Selected Tickers:
{a1.selected_stock_tickers}

### Quant Engine Output:
Expected annual return: {qout.expected_annual_return:.4f}
Annual volatility:       {qout.annual_volatility:.4f}
Optimization details:    {qout.optimization_details}

### CFA Ethical Guidelines:
{cfa_ethical_guidelines_str}

---

### OUTPUT REQUIREMENT (VERY IMPORTANT):
Respond with **ONLY** a JSON object with EXACTLY this schema:

{{
  "suitability_explanation": "text",
  "ethical_considerations": "text",
  "shariah_compliance_statement": "text",
  "risk_assessment": "text",
  "limitations": "text"
}}
"""

    # --- COMPLETION CALL ---
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content":
                "You are a CFA charterholder advisor. Follow CFA ethics and produce strictly JSON outputs."
            },
            {"role": "user", "content": user_prompt}
        ],
    )

    return Agent2Output.model_validate_json(res.choices[0].message.content)



# ============================================================================
# 7 — Cached wrappers & preload
# ============================================================================


@st.cache_resource
def load_quant_engine() -> pd.DataFrame:
    return load_sp500_list()


@st.cache_resource
def run_quant_engine(tickers: Tuple[str, ...]) -> QuantEngineOutput:
    return quant_engine(list(tickers))


def run_agent2_report(p: CustomerProfile, a1: Agent1Output, qout: QuantEngineOutput) -> Agent2Output:
    try:
        with open('cfa_standards_full.json', 'r') as f:
            cfa_data = json.load(f)

        cfa_principles = cfa_data.get("cfa_principles", [])
        cfa_ethical_guidelines_str = "\n".join([f"- {p}" for p in cfa_principles])

    except Exception as e:
        cfa_ethical_guidelines_str = (
            f"CFA standards unavailable ({e}). Ethical considerations will be limited."
        )
    return agent2_gpt(p, a1, qout, cfa_ethical_guidelines_str)


@st.cache_resource
def preload_data() -> str:
    """
    Preload heavy datasets in the background while the user fills the form:
    - S&P 500 universe
    - Historical monthly prices for a sample of tickers
    """
    sp500 = load_sp500_list()
    if "ticker" in sp500.columns:
        tickers = sp500["ticker"].dropna().unique().tolist()
    else:
        tickers = []

    if len(tickers) > 20:
        tickers = tickers[:20]

    if tickers:
        _ = download_prices(tickers)

    return "ok"


# ============================================================================
# 8 — UI Helpers
# ============================================================================


def chat_bubble(title: str, body: str):
    st.markdown(
        f"<div class='chat-bubble'><b>{title}</b><br>{body}</div>",
        unsafe_allow_html=True,
    )


def safe_index(options_list, value):
    """
    Return index of value in options_list, or 0 if value is not present.
    This prevents ValueError when prefilled values don't match the UI options.
    """
    try:
        return options_list.index(value)
    except ValueError:
        return 0


# ============================================================================
# 9 — Client Profile Page (FULL QUESTIONNAIRE)
# ============================================================================


def client_profile_page(sp500: pd.DataFrame):
    # Preload data in the background while user fills the form
    _ = preload_data()
    sample_profile = {
        "full_name": "Future Homeless Millionaire",
        "email": "millionaire@abc.com",

        # Return objectives
        "primary_goal": "Long-term growth",
        "target_annual_return": 8.5,
        "tradeoff_loss_vs_return": 7,
        "tolerance_underperformance": 6,

        # Liquidity & time horizon
        "withdrawal_start": "10+ years",
        "investment_horizon": "> 10 years",
        "desired_liquid_portion_pct": 15,

        # Legal & regulatory constraints
        "legal_constraints": ["Shariah compliance", "UCITS/AIFMD"],
        "other_legal_notes": "None",

        # Tax considerations
        "tax_status": "Corporate",
        "tax_priority": "Balance both",
        "tax_deferral_preference": 7,
        "tax_jurisdictions_to_avoid": "PFICs",

        # ESG
        "esg_exclusions": ["Weapons", "Fossil fuels"],
        "esg_flags": ["Fully Shariah-compliant"],
        "esg_importance": 5,
        "esg_return_tradeoff": 5,

        # Diversification & allocation
        "allocation_equities_pct": 70.0,
        "allocation_fixed_income_pct": 20.0,
        "allocation_alternatives_pct": 10.0,
        "concentration_tolerance": "Moderate",
        "max_single_issuer_exposure_pct": 12.0,
        "management_style": "Hybrid",

        # Performance & reporting
        "preferred_benchmark": "S&P 500",
        "reporting_frequency": "Quarterly",

        # Derived fields
        "risk_tolerance": "medium",         # example value
        "islamic_investing": True           # derived from flags
    }

    st.title("Client Profile & Institutional Investor Questionnaire")
    if st.button("Load Sample Profile"):
        st.session_state["prefill"] = sample_profile
        st.rerun()

    st.markdown(
        "Use this questionnaire to capture your institutional profile, "
        "constraints, and reporting preferences."
    )

    with st.form("client_full_profile_form"):
        pre = st.session_state.get("prefill", {})
        # Ensure all CustomerProfile-required keys exist in prefill dictionary
        required_defaults = {
            "full_name": "",
            "email": "",
            "primary_goal": "Long-term growth",
            "target_annual_return": 5.0,
            "tradeoff_loss_vs_return": 5,
            "tolerance_underperformance": 5,

            "withdrawal_start": "Flexible",
            "investment_horizon": "3–10 years",
            "desired_liquid_portion_pct": 10,

            "legal_constraints": [],
            "other_legal_notes": "",

            "tax_status": "Standard",
            "tax_priority": "Neutral",
            "tax_deferral_preference": 5,
            "tax_jurisdictions_to_avoid": "",

            "esg_exclusions": [],
            "esg_flags": [],
            "esg_importance": 5,
            "esg_return_tradeoff": 5,

            "allocation_equities_pct": 60.0,
            "allocation_fixed_income_pct": 30.0,
            "allocation_alternatives_pct": 10.0,
            "concentration_tolerance": "Moderate",
            "max_single_issuer_exposure_pct": 15.0,
            "management_style": "Hybrid",

            "preferred_benchmark": "Select…",
            "reporting_frequency": "Select…",

            # Derived fields (not shown in UI but must exist in pre)
            "risk_tolerance": "medium",
            "islamic_investing": False,
        }

        for k, v in required_defaults.items():
            pre.setdefault(k, v)


        # ------------------ Client Details ------------------
        st.markdown("### Client Details")
        st.caption("Hint: Tell us who you are so we can label your profile and reports.")

        col_cd1, col_cd2 = st.columns(2)
        with col_cd1:
            full_name = st.text_input(
                "Full name *",
                value=pre.get("full_name", ""),
                placeholder="e.g., Khalid Al Hemaidi",
            )
        with col_cd2:
            email = st.text_input(
                "Email *",
                value=pre.get("email", ""),
                placeholder="name@example.com",
            )

        st.divider()

        # ------------------ 1) Return Objectives ------------------
        st.markdown("### 1) Return Objectives")
        st.caption("Hint: Understand your target returns and trade-offs.")

        primary_options = ["Capital preservation", "Income", "Long-term growth"]
        primary_goal = st.radio(
            "Primary goal *",
            primary_options,
            index=safe_index(primary_options, pre.get("primary_goal", "Long-term growth")),
        )

        target_annual_return = st.number_input(
            "Target annual return (%)",
            min_value=0.0,
            max_value=40.0,
            step=0.1,
            value=pre.get("target_annual_return", 8.0),
        )

        st.markdown("**Trade-off: Minimize losses ↔ Maximize returns**")
        tradeoff_loss_vs_return = st.slider(
            "Trade-off...",
            0,
            10,
            pre.get("tradeoff_loss_vs_return", 5),
        )

        tolerance_underperformance = st.slider(
            "Tolerance for temporary underperformance",
            0,
            10,
            pre.get("tolerance_underperformance", 5),
        )

        st.divider()

        # ------------------ 2) Liquidity & Time Horizon ------------------
        st.markdown("### 2) Liquidity & Time Horizon")
        st.caption("Hint: How long you can invest and how much should stay liquid.")

        col_liq1, col_liq2 = st.columns(2)
        with col_liq1:
            options_list = [
                "Select…",
                "Within 12 months",
                "1–3 years",
                "3–10 years",
                "10+ years",
                "Undecided",
            ]
            withdrawal_start = st.selectbox(
                "When do you expect to start withdrawals?",
                options_list,
                index=safe_index(options_list, pre.get("withdrawal_start", "Select…")),
            )
        with col_liq2:
            options_list = ["Select…", "< 3 years", "3–10 years", "> 10 years"]
            investment_horizon = st.selectbox(
                "Investment horizon *",
                options_list,
                index=safe_index(options_list, pre.get("investment_horizon", "Select…")),
            )

        desired_liquid_portion_pct = st.slider(
            "Desired liquid portion of portfolio (%)",
            0,
            100,
            pre.get("desired_liquid_portion_pct", 20),
        )

        st.divider()

        # ------------------ 3) Legal & Regulatory Constraints ------------------
        st.markdown("### 3) Legal & Regulatory Constraints")
        st.caption("Hint: Compliance boundaries we must respect.")

        st.markdown("**Constraints (Check all that apply)**")
        col_lc1, col_lc2 = st.columns(2)
        with col_lc1:
            lc_shariah = st.checkbox(
                "Shariah compliance",
                value=("Shariah compliance" in pre.get("legal_constraints", [])),
            )
            lc_pension = st.checkbox(
                "Pension/ERISA",
                value=("Pension/ERISA" in pre.get("legal_constraints", [])),
            )
        with col_lc2:
            lc_ucits = st.checkbox(
                "UCITS/AIFMD",
                value=("UCITS/AIFMD" in pre.get("legal_constraints", [])),
            )
            lc_jurisdiction = st.checkbox(
                "Jurisdiction limits",
                value=("Jurisdiction limits" in pre.get("legal_constraints", [])),
            )

        legal_constraints = []
        if lc_shariah:
            legal_constraints.append("Shariah compliance")
        if lc_pension:
            legal_constraints.append("Pension/ERISA")
        if lc_ucits:
            legal_constraints.append("UCITS/AIFMD")
        if lc_jurisdiction:
            legal_constraints.append("Jurisdiction limits")

        other_legal_notes = st.text_area(
            "Other legal/regulatory notes",
            placeholder="Describe any additional restrictions…",
            value=pre.get("other_legal_notes", ""),
        )
        st.caption(
            "Note: We always operate to a fiduciary standard; let us know if your "
            "situation requires anything stricter."
        )

        st.divider()

        # ------------------ 4) Tax Considerations ------------------
        st.markdown("### 4) Tax Considerations")
        st.caption("Hint: Optimize after-tax outcomes.")

        col_tax1, col_tax2 = st.columns(2)
        with col_tax1:
            options_list = [
                "Select…",
                "Individual",
                "Corporate",
                "Foundation/Endowment",
                "Tax-exempt",
                "Other",
            ]
            tax_status = st.selectbox(
                "Tax status *",
                options_list,
                index=safe_index(options_list, pre.get("tax_status", "Select…")),
            )
        with col_tax2:
            options_list = [
                "Select…",
                "Minimize capital gains",
                "Minimize income taxes",
                "Balance both",
                "No preference",
            ]
            tax_priority = st.selectbox(
                "Priority *",
                options_list,
                index=safe_index(options_list, pre.get("tax_priority", "Select…")),
            )

        tax_deferral_preference = st.slider(
            "Preference for tax deferral (buy-and-hold)",
            0,
            10,
            pre.get("tax_deferral_preference", 5),
            help="0 = Low, 10 = High",
        )

        tax_jurisdictions_to_avoid = st.text_input(
            "Jurisdictions/structures to avoid",
            placeholder="e.g., PFICs, certain offshore funds…",
            value=pre.get("tax_jurisdictions_to_avoid", ""),
        )

        st.divider()

        # ------------------ 5) Ethical, Religious, or Social Constraints ------------------
        st.markdown("### 5) Ethical, Religious, or Social Constraints (ESG/SRI)")
        st.caption("Hint: Align the portfolio with your values.")

        st.markdown("**Exclude the following industries (Check all that apply):**")
        col_esg1, col_esg2, col_esg3 = st.columns(3)
        with col_esg1:
            ex_tobacco = st.checkbox(
                "Tobacco", value=("Tobacco" in pre.get("esg_exclusions", []))
            )
            ex_alcohol = st.checkbox(
                "Alcohol", value=("Alcohol" in pre.get("esg_exclusions", []))
            )
        with col_esg2:
            ex_gambling = st.checkbox(
                "Gambling", value=("Gambling" in pre.get("esg_exclusions", []))
            )
            ex_weapons = st.checkbox(
                "Weapons", value=("Weapons" in pre.get("esg_exclusions", []))
            )
        with col_esg3:
            ex_fossil = st.checkbox(
                "Fossil fuels", value=("Fossil fuels" in pre.get("esg_exclusions", []))
            )
            ex_adult = st.checkbox(
                "Adult entertainment",
                value=("Adult entertainment" in pre.get("esg_exclusions", [])),
            )

        esg_exclusions = []
        if ex_tobacco:
            esg_exclusions.append("Tobacco")
        if ex_alcohol:
            esg_exclusions.append("Alcohol")
        if ex_gambling:
            esg_exclusions.append("Gambling")
        if ex_weapons:
            esg_exclusions.append("Weapons")
        if ex_fossil:
            esg_exclusions.append("Fossil fuels")
        if ex_adult:
            esg_exclusions.append("Adult entertainment")

        st.markdown("**ESG flags (Check all that apply):**")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            flag_shariah_full = st.checkbox(
                "Fully Shariah-compliant",
                value=("Fully Shariah-compliant" in pre.get("esg_flags", [])),
            )
        with col_f2:
            flag_positive_esg = st.checkbox(
                "Positive ESG screening",
                value=("Positive ESG screening" in pre.get("esg_flags", [])),
            )
            flag_impact = st.checkbox(
                "Impact investing",
                value=("Impact investing" in pre.get("esg_flags", [])),
            )

        esg_flags = []
        if flag_shariah_full:
            esg_flags.append("Fully Shariah-compliant")
        if flag_positive_esg:
            esg_flags.append("Positive ESG screening")
        if flag_impact:
            esg_flags.append("Impact investing")

        esg_importance = st.slider(
            "Importance of ESG alignment",
            0,
            10,
            pre.get("esg_importance", 5),
            help="0 = Low, 10 = High",
        )

        esg_return_tradeoff = st.slider(
            "Willingness to trade returns for ethics",
            0,
            10,
            pre.get("esg_return_tradeoff", 5),
            help="0 = Not willing, 10 = Very willing",
        )

        st.divider()

        # ------------------ 6) Diversification & Allocation Rules ------------------
        st.markdown("### 6) Diversification & Allocation Rules")
        st.caption("Hint: Preferred asset mix and concentration limits.")

        st.markdown("**Asset Allocation (%): (Total should equal 100%)**")
        col_alloc1, col_alloc2, col_alloc3 = st.columns(3)
        with col_alloc1:
            allocation_equities_pct = st.number_input(
                "Equities (%)",
                min_value=0.0,
                max_value=100.0,
                value=pre.get("allocation_equities_pct", 60.0),
            )
        with col_alloc2:
            allocation_fixed_income_pct = st.number_input(
                "Fixed Income (%)",
                min_value=0.0,
                max_value=100.0,
                value=pre.get("allocation_fixed_income_pct", 30.0),
            )
        with col_alloc3:
            allocation_alternatives_pct = st.number_input(
                "Alternatives (%)",
                min_value=0.0,
                max_value=100.0,
                value=pre.get("allocation_alternatives_pct", 10.0),
            )

        concentration_tolerance = st.radio(
            "Concentration tolerance *",
            ["Low", "Moderate", "High"],
            index=safe_index(["Low", "Moderate", "High"], pre.get("concentration_tolerance", "Moderate")),
        )

        max_single_issuer_exposure_pct = st.number_input(
            "Max exposure to a single issuer/sector (%)",
            min_value=0.0,
            max_value=100.0,
            value=pre.get("max_single_issuer_exposure_pct", 15.0),
        )

        management_style = st.radio(
            "Management style preference *",
            ["Passive", "Active", "Hybrid"],
            index=safe_index(["Passive", "Active", "Hybrid"], pre.get("management_style", "Hybrid")),
        )

        st.divider()

        # ------------------ 7) Performance & Reporting ------------------
        st.markdown("### 7) Performance & Reporting")
        st.caption("Hint: Benchmarks, report cadence, and transparency.")

        col_rep1, col_rep2 = st.columns(2)
        with col_rep1:
            options_list = [
                "Select…",
                "S&P 500",
                "MSCI World",
                "MSCI ACWI IMI (ESG)",
                "Custom (describe below)",
            ]
            preferred_benchmark = st.selectbox(
                "Preferred benchmark",
                options_list,
                index=safe_index(options_list, pre.get("preferred_benchmark", "Select…")),
            )
        with col_rep2:
            options_list = ["Select…", "Monthly", "Quarterly", "Semi-annual", "Annual"]
            reporting_frequency = st.selectbox(
                "Reporting frequency *",
                options_list,
                index=safe_index(options_list, pre.get("reporting_frequency", "Select…")),
            )

        custom_benchmark_notes = st.text_input(
            "If custom benchmark, briefly describe",
            value=pre.get("custom_benchmark_notes", ""),
        )

        # ------------------ Submit ------------------
        submitted = st.form_submit_button("Submit Profile & Generate Portfolio")

    # ------------------ Handle Submit ------------------
    if submitted:
        errors = []

        if not full_name.strip():
            errors.append("Full name is required.")
        if not email.strip():
            errors.append("Email is required.")
        if investment_horizon == "Select…":
            errors.append("Investment horizon is required.")
        if tax_status == "Select…":
            errors.append("Tax status is required.")
        if tax_priority == "Select…":
            errors.append("Tax priority is required.")
        if reporting_frequency == "Select…":
            errors.append("Reporting frequency is required.")

        # Asset allocation sum check
        alloc_sum = (
            allocation_equities_pct
            + allocation_fixed_income_pct
            + allocation_alternatives_pct
        )
        if abs(alloc_sum - 100.0) > 1e-6:
            errors.append(
                f"Asset allocation must sum to 100%. Current total: {alloc_sum:.1f}%."
            )

        if errors:
            st.error(
                "Please fix the following before continuing:\n- " + "\n- ".join(errors)
            )
            return

        # Derived risk_tolerance
        # Base on tradeoff slider + concentration preference
        base_risk_score = tradeoff_loss_vs_return
        if concentration_tolerance == "Low":
            base_risk_score -= 2
        elif concentration_tolerance == "High":
            base_risk_score += 2
        base_risk_score = max(0, min(10, base_risk_score))

        if base_risk_score <= 3:
            risk_tolerance = "low"
        elif base_risk_score <= 7:
            risk_tolerance = "medium"
        else:
            risk_tolerance = "high"

        # Derived Islamic investing flag
        islamic_investing = bool(lc_shariah or flag_shariah_full)

        # Merge custom benchmark description if relevant
        if preferred_benchmark == "Custom (describe below)" and custom_benchmark_notes:
            preferred_benchmark_value = f"Custom: {custom_benchmark_notes}"
        else:
            preferred_benchmark_value = preferred_benchmark

        profile = CustomerProfile(
            full_name=full_name.strip(),
            email=email.strip(),
            primary_goal=primary_goal,
            target_annual_return=float(target_annual_return),
            tradeoff_loss_vs_return=int(tradeoff_loss_vs_return),
            tolerance_underperformance=int(tolerance_underperformance),
            withdrawal_start=withdrawal_start,
            investment_horizon=investment_horizon,
            desired_liquid_portion_pct=int(desired_liquid_portion_pct),
            legal_constraints=legal_constraints,
            other_legal_notes=other_legal_notes.strip(),
            tax_status=tax_status,
            tax_priority=tax_priority,
            tax_deferral_preference=int(tax_deferral_preference),
            tax_jurisdictions_to_avoid=tax_jurisdictions_to_avoid.strip(),
            esg_exclusions=esg_exclusions,
            esg_flags=esg_flags,
            esg_importance=int(esg_importance),
            esg_return_tradeoff=int(esg_return_tradeoff),
            allocation_equities_pct=float(allocation_equities_pct),
            allocation_fixed_income_pct=float(allocation_fixed_income_pct),
            allocation_alternatives_pct=float(allocation_alternatives_pct),
            concentration_tolerance=concentration_tolerance,
            max_single_issuer_exposure_pct=float(max_single_issuer_exposure_pct),
            management_style=management_style,
            preferred_benchmark=preferred_benchmark_value,
            reporting_frequency=reporting_frequency,
            risk_tolerance=risk_tolerance,
            islamic_investing=islamic_investing,
        )

        # Save full questionnaire to session_state
        st.session_state["profile"] = profile
        st.session_state["client_full_profile"] = profile.model_dump()

        # Clear any previous downstream computations
        if "a1" in st.session_state:
            del st.session_state["a1"]
        if "report" in st.session_state:
            del st.session_state["report"]

        # IMPORTANT CHANGE:
        # Do NOT run Agent 1 here.
        # Just redirect to the Portfolio Recommendation page.
        st.session_state["current_page"] = "Portfolio Recommendation"
        st.rerun()


# ============================================================================
# 10 — Portfolio Recommendation Page
# ============================================================================


def portfolio_page(sp500: pd.DataFrame):
    st.title("Portfolio Recommendation")

    # Need a profile first
    if "profile" not in st.session_state:
        st.info("Please complete the Client Profile questionnaire first.")
        return

    profile: CustomerProfile = st.session_state["profile"]

    # IMPORTANT CHANGE:
    # Agent 1 MUST run only on this page (not on the Client Profile page).
    if "a1" not in st.session_state:
        time.sleep(1)
        with st.spinner("Running Agent 1 (stock selection)…"):
            st.session_state["a1"] = agent1_gpt4o(profile, sp500)

    a1: Agent1Output = st.session_state["a1"]

    # ----------------------------------------------------------------------
    # 1) Agent 1 output (tickers + reasoning) — MUST BE FIRST
    # ----------------------------------------------------------------------
    st.markdown(
        '<div class="section-header">Agent 1 — Stock Selection</div>',
        unsafe_allow_html=True,
    )

    tickers_str = (
        ", ".join(a1.selected_stock_tickers)
        if a1.selected_stock_tickers
        else "No tickers returned."
    )
    st.markdown(
        f"<div class='chat-bubble'><b>Selected Tickers:</b> {tickers_str}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='chat-bubble'>{a1.reasoning}</div>",
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------------
    # Optional Client Summary (still near the top, but after Agent 1)
    # ----------------------------------------------------------------------
    st.markdown(
        '<div class="section-header">Client Summary</div>', unsafe_allow_html=True
    )
    st.markdown(
        f"""
<div class="client-card">
<b>Name:</b> {profile.full_name}<br>
<b>Primary goal:</b> {profile.primary_goal}<br>
<b>Target return:</b> {profile.target_annual_return:.1f}%<br>
<b>Horizon:</b> {profile.investment_horizon}<br>
<b>Liquidity need:</b> {profile.desired_liquid_portion_pct}% liquid<br>
<b>Risk tolerance (derived):</b> {profile.risk_tolerance.title()}<br>
<b>Management style:</b> {profile.management_style}<br>
<b>Benchmark preference:</b> {profile.preferred_benchmark}<br>
<b>Reporting:</b> {profile.reporting_frequency}
</div>
""",
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------------
    # Run Quant Engine
    # ----------------------------------------------------------------------
    ticktuple = tuple(a1.selected_stock_tickers)
    if not ticktuple:
        st.warning("Agent 1 did not return any tickers to optimize.")
        return

    with st.spinner("Running Quant Engine…"):
        qout = run_quant_engine(ticktuple)

    # ----------------------------------------------------------------------
    # 2) KPI cards (expected return, volatility)
    # ----------------------------------------------------------------------
    st.markdown(
        '<div class="section-header">Expected Return & Volatility</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>Expected Annual Return</div>"
        f"<div class='kpi-value'>{qout.expected_annual_return:.2%}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>Annual Volatility</div>"
        f"<div class='kpi-value'>{qout.annual_volatility:.2%}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------------
    # 3) Enhanced allocation pie chart (Top 10 + Others) & 4) Allocation table
    # ----------------------------------------------------------------------
    st.markdown(
        '<div class="section-header">Allocation</div>', unsafe_allow_html=True
    )
    with st.spinner("Generating allocation charts…"):
        time.sleep(2)
        if qout.portfolio_weights:
            dfw = pd.DataFrame(
                {
                    "Ticker": list(qout.portfolio_weights.keys()),
                    "Weight": list(qout.portfolio_weights.values()),
                }
            ).sort_values("Weight", ascending=False)

            # Top 10 + Others
            top_n = min(10, len(dfw))
            top = dfw.head(top_n).copy()
            others_weight = dfw["Weight"].iloc[top_n:].sum()
            if others_weight > 0:
                top = pd.concat(
                    [top, pd.DataFrame([{"Ticker": "Others", "Weight": others_weight}])],
                    ignore_index=True,
                )

            # Two-column layout (Pie + Table)
            col_pie, col_table = st.columns([2, 1])

            with col_pie:
                fig_pie = px.pie(
                    top,
                    names="Ticker",
                    values="Weight",
                    hole=0.4,
                )
                fig_pie.update_traces(
                    textinfo="label+percent",
                    hovertemplate="%{label}<br>Weight=%{value:.0%}<extra></extra>",
                )
                fig_pie.update_layout(
                    margin=dict(l=0, r=0, t=20, b=0),
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, width="stretch")

            with col_table:
                name_map = None
                if {"ticker", "name"}.issubset(sp500.columns):
                    name_map = sp500[["ticker", "name"]].rename(
                        columns={"ticker": "Ticker", "name": "Name"}
                    )
                df_table = dfw.copy()
                if name_map is not None:
                    df_table = df_table.merge(name_map, on="Ticker", how="left")
                else:
                    df_table["Name"] = ""

                df_table = df_table[["Ticker", "Name", "Weight"]].sort_values(
                    "Weight", ascending=False
                )
                df_table["Weight"] = df_table["Weight"].map(lambda x: f"{x:.0%}")
                st.markdown("**Full Allocation**")
                st.dataframe(df_table, width="stretch", height=400)
        else:
            st.warning("No portfolio weights available.")

    # ----------------------------------------------------------------------
    # 5) 5-year backtest vs S&P 500 chart (percentage axis)
    # ----------------------------------------------------------------------
    st.markdown(
        '<div class="section-header">Portfolio Backtest — 5-Year Performance vs S&P 500</div>',
        unsafe_allow_html=True,
    )
    with st.spinner("Running 5-year backtest…"):
        time.sleep(1)
        try:
            price_data = download_prices(list(qout.portfolio_weights.keys()))

            sp500_data = yf.download(
                "^GSPC",
                period="5y",
                interval="1mo",
                progress=False,
                auto_adjust=False,
                multi_level_index=False,
            )

            if (
                not price_data.empty
                and len(price_data.columns) >= 2
                and not sp500_data.empty
            ):
                # Normalize portfolio to 1.0 at start
                norm_prices = price_data / price_data.iloc[0]
                weights_series = pd.Series(qout.portfolio_weights)
                weights_series = weights_series.reindex(norm_prices.columns).fillna(0.0)
                portfolio_value = (norm_prices * weights_series).sum(axis=1)

                # S&P 500 normalized
                sp500_series = sp500_data["Adj Close"]
                sp500_series = sp500_series.reindex(portfolio_value.index, method="ffill")
                sp500_norm = sp500_series / sp500_series.iloc[0]

                # Convert to % change (start = 0%)
                portfolio_pct = (portfolio_value - 1.0) * 100.0
                sp500_pct = (sp500_norm - 1.0) * 100.0

                backtest_df = pd.DataFrame(
                    {
                        "Portfolio": portfolio_pct,
                        "S&P 500": sp500_pct,
                    }
                )

                fig_backtest = px.line(
                    backtest_df,
                    labels={"index": "Date", "value": "Return (%)", "variable": ""},
                    title="Backtested Portfolio Performance vs S&P 500 (5 Years)",
                )
                fig_backtest.update_layout(
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    ),
                )
                st.plotly_chart(fig_backtest, width="stretch")
            else:
                st.warning("Insufficient data to run the backtest.")
        except Exception as e:
            st.warning(f"Could not compute backtest: {e}")

    # ----------------------------------------------------------------------
    # 6) Industry sector treemap
    # ----------------------------------------------------------------------
    with st.spinner("Building sector allocation treemap…"):
        time.sleep(1)
        ticker_to_sector = sp500.set_index('ticker')['sector'].to_dict() # Assuming 'sp500' has 'ticker' and 'Sector' columns

        ind = []
        for t, w in qout.portfolio_weights.items():
            sector = ticker_to_sector.get(t, "Unknown") # Get sector, default to "Unknown" if not found
            ind.append({"sector": sector, "Weight": w})
            
        ind_df = pd.DataFrame(ind).groupby("sector", as_index=False)["Weight"].sum()
        
        # Sort the data for better visualization flow
        ind_df = ind_df.sort_values(by='Weight', ascending=False)
        
        fig_tree = px.treemap(
            ind_df, 
            path=[px.Constant("Portfolio"), 'sector'],
            values='Weight',
            color='Weight',
            color_continuous_scale='Blues', # Other good options: 'Jet', 'Rainbow', 'HSV', 'Turbo'
        )

        # Customize the text and layout
        fig_tree.update_traces(
            textinfo="label+percent parent",
            textfont_size=16, # Adjust this value as needed
            textfont=dict(weight='bold'),
            hovertemplate="Sector: %{label}<br>Weight: %{value:.1%}<extra></extra>"
        )
        
        fig_tree.update_layout(
            margin = dict(t=25, l=0, r=0, b=0),
            title_text='Sector Allocation Treemap',
            # Optional: You can also increase the overall font size for the title if needed
            # title_font_size=24 
        )

        st.plotly_chart(fig_tree, width="stretch")

    # ----------------------------------------------------------------------
    # 7) Agent 2 narrative report
    # ----------------------------------------------------------------------
    st.markdown(
        '<div class="section-header">Agent 2 — Narrative Report</div>',
        unsafe_allow_html=True,
    )

    if "report" not in st.session_state:
        with st.spinner("Generating Agent 2 report…"):
            st.session_state["report"] = run_agent2_report(profile, a1, qout)

    rep: Agent2Output = st.session_state["report"]

    chat_bubble("Suitability", rep.suitability_explanation)
    chat_bubble("Ethical & ESG Considerations", rep.ethical_considerations)
    chat_bubble("Shariah Compliance", rep.shariah_compliance_statement)
    chat_bubble("Risk Assessment", rep.risk_assessment)
    chat_bubble("Limitations", rep.limitations)


# ============================================================================
# 11 — MAIN
# ============================================================================


def main():
    st.sidebar.title("Navigation")

    # Ensure current_page has a default
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Client Profile"

    page_options = ["Client Profile", "Portfolio Recommendation"]

    # Determine which page should be highlighted based on current_page
    current = st.session_state["current_page"]
    if current not in page_options:
        current = "Client Profile"

    default_index = page_options.index(current)

    # Sidebar widget WITHOUT a key so we never mutate a widget key on redirect
    page = st.sidebar.radio(
        "",
        page_options,
        index=default_index,
    )

    # Sync selected page into session_state
    if page != st.session_state["current_page"]:
        st.session_state["current_page"] = page

    sp500 = load_quant_engine()

    if st.session_state["current_page"] == "Client Profile":
        client_profile_page(sp500)
    else:
        portfolio_page(sp500)


if __name__ == "__main__":
    main()
