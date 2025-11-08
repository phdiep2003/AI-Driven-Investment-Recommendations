# app.py
from flask import Flask, request, render_template_string, send_from_directory, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os, json, re

app = Flask(__name__)
app.secret_key = "replace-with-a-random-secret"  # for flash messages

# ---------- Database config ----------
# Use SQLite by default; switch to Postgres by setting DATABASE_URL env var:
# export DATABASE_URL="postgresql+psycopg://user:pass@host:5432/dbname"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///submissions.db")
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ---------- Model ----------
class Submission(db.Model):
    __tablename__ = "submissions"
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    # handy searchable columns
    full_name = db.Column(db.String(200), index=True)
    email = db.Column(db.String(200), index=True)
    goal = db.Column(db.String(80), index=True)            # Capital Preservation / Income / Long-Term Growth
    horizon = db.Column(db.String(40), index=True)         # <3 / 3–10 / >10
    style = db.Column(db.String(40), index=True)           # Passive / Active / Hybrid
    benchmark = db.Column(db.String(120), index=True)
    freq = db.Column(db.String(40), index=True)
    # allocations as numbers
    alloc_equities = db.Column(db.Float)
    alloc_fixed = db.Column(db.Float)
    alloc_alts = db.Column(db.Float)
    # full JSON payload
    payload_json = db.Column(db.Text)

with app.app_context():
    db.create_all()

# ---------- helpers ----------
def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s)
    s = re.sub(r"^-+|-+$", "", s)
    return s or "client"

def to_number(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)

def to_float(x, default=0.0):
    """Defensive float cast."""
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str) and x.strip() != "":
            return float(x)
        return float(default)
    except Exception:
        return float(default)

def build_payload(form):
    data = {k: v for k, v in form.items()}
    payload = {
        "client": {"fullName": data.get("fullName"), "email": data.get("email")},
        "returnObjectives": {
            "goal": data.get("goal"),
            "targetReturnPct": data.get("targetReturn"),
            "riskReturnTradeoff_0to10": to_number(data.get("riskTradeoff"), 5),
            "underperformanceTolerance_0to10": to_number(data.get("underperfTolerance"), 5),
        },
        "liquidityAndHorizon": {
            "withdrawStart": data.get("withdrawStart"),
            "horizon": data.get("horizon"),
            "liquidPct": to_number(data.get("liquidPct"), 0),
        },
        "legalAndRegulatory": {
            "flags": request.form.getlist("legal"),
            "notes": data.get("legalNotes", ""),
        },
        "tax": {
            "status": data.get("taxStatus"),
            "priority": data.get("taxPriority"),
            "deferralPreference_0to10": to_number(data.get("taxDeferral"), 0),
            "avoid": data.get("taxAvoid", ""),
        },
        "esgSRI": {
            "exclude": request.form.getlist("esgExclude"),
            "flags": request.form.getlist("esgFlags"),
            "importance_0to10": to_number(data.get("esgImportance"), 0),
            "returnTradeoffWillingness_0to10": to_number(data.get("esgTradeoff"), 0),
        },
        "diversificationAndAllocation": {
            "targetAllocationPct": {
                "equities": to_number(data.get("allocEquities"), 0),
                "fixedIncome": to_number(data.get("allocFixed"), 0),
                "alternatives": to_number(data.get("allocAlts"), 0),
            },
            "concentrationTolerance": data.get("concentration"),
            "maxExposurePct": to_number(data.get("maxExposure"), 0) if data.get("maxExposure") else None,
            "style": data.get("style"),
        },
        "performanceAndReporting": {
            "benchmark": data.get("benchmark"),
            "frequency": data.get("freq"),
            "detailLevel": data.get("detail"),
            "feeTransparencyImportance_0to10": to_number(data.get("feeTransparency"), 0),
            "notes": data.get("perfNotes", ""),
        },
        "meta": {
            "submittedAt": datetime.utcnow().isoformat() + "Z",
            "userAgent": request.headers.get("User-Agent", ""),
            "remoteAddr": request.remote_addr,
        },
    }
    return payload

# ---------- routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(TPL, saved=None, json_text=None)

@app.route("/submit", methods=["POST"])
def submit():
    errors = []

    # Required fields
    required_fields = ["fullName", "email", "horizon", "taxStatus", "taxPriority", "freq", "detail"]
    missing = [f for f in required_fields if not request.form.get(f)]
    if missing:
        errors.append("Please complete required fields: " + ", ".join(missing))

    for group in ["goal", "concentration", "style"]:
        if not request.form.get(group):
            errors.append(f"Please choose an option for: {group}")

    # Allocation validation
    eq = to_number(request.form.get("allocEquities"), 0)
    fx = to_number(request.form.get("allocFixed"), 0)
    al = to_number(request.form.get("allocAlts"), 0)
    alloc_sum = round(eq + fx + al, 2)
    if alloc_sum != 100.0:
        errors.append(f"Asset allocation must total 100%. Currently: {alloc_sum}%")

    if errors:
        for e in errors:
            flash(e, "error")
        return render_template_string(TPL, saved=None, json_text=None), 400

    payload = build_payload(request.form)
    tap = payload["diversificationAndAllocation"]["targetAllocationPct"]  # dict with three floats

    # ---- Persist to DB ----
    sub = Submission(
        full_name = payload["client"]["fullName"] or "",
        email     = payload["client"]["email"] or "",
        goal      = payload["returnObjectives"]["goal"] or "",
        horizon   = payload["liquidityAndHorizon"]["horizon"] or "",
        style     = payload["diversificationAndAllocation"]["style"] or "",
        benchmark = payload["performanceAndReporting"]["benchmark"] or "",
        freq      = payload["performanceAndReporting"]["frequency"] or "",
        alloc_equities = to_float(tap.get("equities", 0)),
        alloc_fixed    = to_float(tap.get("fixedIncome", 0)),
        alloc_alts     = to_float(tap.get("alternatives", 0)),
        payload_json   = json.dumps(payload, ensure_ascii=False)
    )
    db.session.add(sub)
    db.session.commit()

    # Optional: also keep a JSON file copy
    os.makedirs("submissions", exist_ok=True)
    fname = f"{sub.created_at.strftime('%Y%m%d-%H%M%S')}_{slugify(sub.full_name)}_{sub.id}.json"
    with open(os.path.join("submissions", fname), "w", encoding="utf-8") as fh:
        fh.write(sub.payload_json)

    flash("Submission saved to database.", "success")
    return render_template_string(TPL, saved=fname, json_text=json.dumps(payload, indent=2, ensure_ascii=False))

@app.route("/download/<filename>", methods=["GET"])
def download_json(filename):
    return send_from_directory("submissions", filename, as_attachment=True)

@app.route("/admin", methods=["GET"])
def admin():
    rows = Submission.query.order_by(Submission.created_at.desc()).limit(50).all()
    return render_template_string(ADMIN_TPL, rows=rows)

# ---------- Templates ----------
TPL = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Client Investment Mandate & Risk Questionnaire</title>
  <style>
    :root{
      --bg:#ffffff; --fg:#0b0f12; --muted:#6b7280;
      --blue:#2563eb; --blue-600:#1d4ed8; --blue-100:#e0e7ff;
      --border:#e5e7eb; --card:#ffffff; --radius:14px;
      --shadow:0 6px 24px rgba(0,0,0,.08); --focus:0 0 0 3px rgba(37,99,235,.35);
    }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Helvetica Neue",Arial,sans-serif;}
    header{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.9);backdrop-filter:saturate(180%) blur(10px);border-bottom:1px solid var(--border);}
    .wrap{max-width:980px;margin:0 auto;padding:18px 20px;}
    h1{margin:0;font-size:1.4rem;font-weight:700;letter-spacing:.2px}
    main{padding:28px 20px;}
    .section{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);padding:20px;margin:16px 0;}
    .section h2{margin:0 0 6px;font-size:1.1rem}
    .section p.hint{margin:4px 0 14px;color:var(--muted);font-size:.95rem}
    fieldset{border:none;padding:0;margin:0}
    .grid{display:grid;gap:14px}
    @media (min-width:720px){ .grid-2{grid-template-columns:1fr 1fr} .grid-3{grid-template-columns:repeat(3,1fr)} }
    label{display:block;font-weight:600;margin-bottom:6px}
    .control{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
    input[type="text"], input[type="number"], input[type="email"], select, textarea{
      width:100%; border:1px solid var(--border); border-radius:10px; padding:10px 12px; outline:none; background:#fff; transition:.15s border-color ease;
    }
    textarea{min-height:90px; resize:vertical}
    input:focus, select:focus, textarea:focus{border-color:var(--blue); box-shadow:var(--focus)}
    .chip{display:inline-flex; align-items:center; gap:8px; border:1px solid var(--border); padding:8px 10px; border-radius:999px}
    .range-wrap{display:grid; gap:6px}
    .range-meta{display:flex; justify-content:space-between; font-size:.9rem; color:var(--muted)}
    input[type="range"]{width:100%}
    .note{font-size:.88rem;color:var(--muted)}
    .bar{height:10px;background:var(--blue-100);border-radius:999px;overflow:hidden;border:1px solid var(--border)}
    .bar > span{display:block;height:100%;background:var(--blue);width:0%}
    .btn-row{display:flex;gap:12px;flex-wrap:wrap;justify-content:flex-end;margin-top:12px}
    button{border:none;cursor:pointer;font-weight:700;padding:10px 16px;border-radius:12px;transition:.15s transform ease,.15s background-color ease}
    .btn-primary{background:var(--blue);color:#fff}.btn-primary:hover{background:var(--blue-600);transform:translateY(-1px)}
    .alert{border:1px solid var(--border);border-radius:12px;padding:12px 14px;margin:12px 0}
    .alert.error{border-color:#fecaca;background:#fff1f2}
    .alert.success{border-color:#bbf7d0;background:#f0fdf4}
    .json-out{white-space:pre-wrap;word-break:break-word;font-family:ui-monospace,Menlo,Consolas,monospace;background:#0b1220;color:#e5e7eb;padding:16px;border-radius:12px;border:1px solid #111827;overflow:auto;max-height:420px}
    .footer-note{color:var(--muted);font-size:.85rem;margin-top:8px}
    .topnav a{margin-right:12px; color:var(--blue); text-decoration:none}
    .topnav a:hover{text-decoration:underline}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>Client Investment Mandate & Risk Questionnaire</h1>
    </div>
  </header>

  <main>
    <div class="wrap">
      {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
          {% for cat, msg in messages %}
            <div class="alert {{cat}}">{{ msg }}</div>
          {% endfor %}
        {% endif %}
      {% endwith %}

      <form id="qForm" method="post" action="/submit" novalidate>
        <!-- Client Details -->
        <div class="section">
          <h2>Client Details</h2>
          <p class="hint">Tell us who you are so we can label your profile and reports.</p>
          <div class="grid grid-2">
            <div>
              <label for="fullName">Full name *</label>
              <input id="fullName" name="fullName" type="text" placeholder="e.g., Khalid Al Hemaidi" required />
            </div>
            <div>
              <label for="email">Email *</label>
              <input id="email" name="email" type="email" placeholder="name@example.com" required />
            </div>
          </div>
        </div>

        <!-- 1. Return Objectives -->
        <div class="section">
          <h2>1) Return Objectives</h2>
          <p class="hint">Understand your target returns and trade-offs.</p>
          <fieldset class="grid grid-2">
            <div>
              <label>Primary goal *</label>
              <div class="control">
                <label class="chip"><input type="radio" name="goal" value="Capital Preservation" required /> Capital preservation</label>
                <label class="chip"><input type="radio" name="goal" value="Income Generation" /> Income</label>
                <label class="chip"><input type="radio" name="goal" value="Long-Term Growth" /> Long-term growth</label>
              </div>
            </div>
            <div>
              <label for="targetReturn">Target annual return (%)</label>
              <input id="targetReturn" name="targetReturn" type="number" min="0" max="40" step="0.1" placeholder="e.g., 8.0" />
            </div>
          </fieldset>

          <div class="grid">
            <div class="range-wrap">
              <label for="riskTradeoff">Trade-off: Minimize losses ↔ Maximize returns</label>
              <input id="riskTradeoff" name="riskTradeoff" type="range" min="0" max="10" value="5" />
              <div class="range-meta"><span>Loss aversion</span><span id="riskTradeoffVal">5/10</span><span>Return seeking</span></div>
            </div>
            <div class="range-wrap">
              <label for="underperfTolerance">Tolerance for temporary underperformance</label>
              <input id="underperfTolerance" name="underperfTolerance" type="range" min="0" max="10" value="5" />
              <div class="range-meta"><span>Low</span><span id="underperfToleranceVal">5/10</span><span>High</span></div>
            </div>
          </div>
        </div>

        <!-- 2. Liquidity & Time Horizon -->
        <div class="section">
          <h2>2) Liquidity & Time Horizon</h2>
          <p class="hint">How long you can invest and how much should stay liquid.</p>
          <fieldset class="grid grid-2">
            <div>
              <label for="withdrawStart">When do you expect to start withdrawals?</label>
              <select id="withdrawStart" name="withdrawStart">
                <option value="">Select…</option>
                <option>Within 12 months</option>
                <option>1–3 years</option>
                <option>3–10 years</option>
                <option>10+ years</option>
                <option>Undecided</option>
              </select>
            </div>
            <div>
              <label for="horizon">Investment horizon *</label>
              <select id="horizon" name="horizon" required>
                <option value="">Select…</option>
                <option>&lt; 3 years</option>
                <option>3–10 years</option>
                <option>&gt; 10 years</option>
              </select>
            </div>
          </fieldset>
          <div class="grid">
            <div class="range-wrap">
              <label for="liquidPct">Desired liquid portion of portfolio (%)</label>
              <input id="liquidPct" name="liquidPct" type="range" min="0" max="100" value="20" />
              <div class="range-meta"><span>0%</span><span id="liquidPctVal">20%</span><span>100%</span></div>
            </div>
          </div>
        </div>

        <!-- 3. Legal & Regulatory -->
        <div class="section">
          <h2>3) Legal & Regulatory Constraints</h2>
          <p class="hint">Compliance boundaries we must respect.</p>
          <fieldset class="grid">
            <div class="control">
              <label class="chip"><input type="checkbox" name="legal" value="Shariah" /> Shariah compliance</label>
              <label class="chip"><input type="checkbox" name="legal" value="Pension/ERISA" /> Pension/ERISA</label>
              <label class="chip"><input type="checkbox" name="legal" value="UCITS/AIFMD" /> UCITS/AIFMD</label>
              <label class="chip"><input type="checkbox" name="legal" value="Jurisdiction Limits" /> Jurisdiction limits</label>
            </div>
            <div>
              <label for="legalNotes">Other legal/regulatory notes</label>
              <textarea id="legalNotes" name="legalNotes" placeholder="Describe any additional restrictions…"></textarea>
            </div>
            <div class="note">We always operate to a fiduciary standard; let us know if your situation requires anything stricter.</div>
          </fieldset>
        </div>

        <!-- 4. Tax Considerations -->
        <div class="section">
          <h2>4) Tax Considerations</h2>
          <p class="hint">Optimize after-tax outcomes.</p>
          <div class="grid grid-2">
            <div>
              <label for="taxStatus">Tax status *</label>
              <select id="taxStatus" name="taxStatus" required>
                <option value="">Select…</option>
                <option>Individual</option>
                <option>Corporate</option>
                <option>Foundation/Endowment</option>
                <option>Tax-exempt</option>
                <option>Other</option>
              </select>
            </div>
            <div>
              <label for="taxPriority">Priority *</label>
              <select id="taxPriority" name="taxPriority" required>
                <option value="">Select…</option>
                <option>Minimize capital gains</option>
                <option>Minimize income taxes</option>
                <option>Balance both</option>
                <option>No preference</option>
              </select>
            </div>
          </div>
          <div class="grid">
            <div class="range-wrap">
              <label for="taxDeferral">Preference for tax deferral (buy-and-hold)</label>
              <input id="taxDeferral" name="taxDeferral" type="range" min="0" max="10" value="6" />
              <div class="range-meta"><span>Low</span><span id="taxDeferralVal">6/10</span><span>High</span></div>
            </div>
            <div>
              <label for="taxAvoid">Jurisdictions/structures to avoid</label>
              <input id="taxAvoid" name="taxAvoid" type="text" placeholder="e.g., PFICs, certain offshore funds…" />
            </div>
          </div>
        </div>

        <!-- 5. ESG / SRI -->
        <div class="section">
          <h2>5) Ethical, Religious, or Social Constraints (ESG/SRI)</h2>
          <p class="hint">Align the portfolio with your values.</p>
          <fieldset class="grid">
            <label>Exclude the following industries:</label>
            <div class="control">
              <label class="chip"><input type="checkbox" name="esgExclude" value="Tobacco" /> Tobacco</label>
              <label class="chip"><input type="checkbox" name="esgExclude" value="Alcohol" /> Alcohol</label>
              <label class="chip"><input type="checkbox" name="esgExclude" value="Gambling" /> Gambling</label>
              <label class="chip"><input type="checkbox" name="esgExclude" value="Weapons" /> Weapons</label>
              <label class="chip"><input type="checkbox" name="esgExclude" value="Fossil Fuels" /> Fossil fuels</label>
              <label class="chip"><input type="checkbox" name="esgExclude" value="Adult Entertainment" /> Adult entertainment</label>
            </div>
            <div class="control">
              <label class="chip"><input type="checkbox" name="esgFlags" value="Shariah-Only" /> Fully Shariah-compliant</label>
              <label class="chip"><input type="checkbox" name="esgFlags" value="Positive Screening" /> Positive ESG screening</label>
              <label class="chip"><input type="checkbox" name="esgFlags" value="Impact Focus" /> Impact investing</label>
            </div>
            <div class="range-wrap">
              <label for="esgImportance">Importance of ESG alignment</label>
              <input id="esgImportance" name="esgImportance" type="range" min="0" max="10" value="7" />
              <div class="range-meta"><span>Low</span><span id="esgImportanceVal">7/10</span><span>High</span></div>
            </div>
            <div class="range-wrap">
              <label for="esgTradeoff">Willingness to trade returns for ethics</label>
              <input id="esgTradeoff" name="esgTradeoff" type="range" min="0" max="10" value="3" />
              <div class="range-meta"><span>Not willing</span><span id="esgTradeoffVal">3/10</span><span>Very willing</span></div>
            </div>
          </fieldset>
        </div>

        <!-- 6. Diversification & Allocation -->
        <div class="section">
          <h2>6) Diversification & Allocation Rules</h2>
          <p class="hint">Preferred asset mix and concentration limits.</p>
          <div class="grid grid-3">
            <div>
              <label for="allocEquities">Equities (%)</label>
              <input id="allocEquities" name="allocEquities" type="number" min="0" max="100" value="60" />
            </div>
            <div>
              <label for="allocFixed">Fixed Income (%)</label>
              <input id="allocFixed" name="allocFixed" type="number" min="0" max="100" value="30" />
            </div>
            <div>
              <label for="allocAlts">Alternatives (%)</label>
              <input id="allocAlts" name="allocAlts" type="number" min="0" max="100" value="10" />
            </div>
          </div>
          <div class="bar" aria-hidden="true" style="margin-top:10px"><span id="allocBar"></span></div>
          <div class="range-meta"><span>Total</span><span id="allocTotal">100%</span><span>Must equal 100%</span></div>

          <div class="grid grid-2" style="margin-top:10px">
            <div>
              <label>Concentration tolerance *</label>
              <div class="control">
                <label class="chip"><input type="radio" name="concentration" value="Low" required /> Low</label>
                <label class="chip"><input type="radio" name="concentration" value="Moderate" /> Moderate</label>
                <label class="chip"><input type="radio" name="concentration" value="High" /> High</label>
              </div>
            </div>
            <div>
              <label for="maxExposure">Max exposure to a single issuer/sector (%)</label>
              <input id="maxExposure" name="maxExposure" type="number" min="0" max="100" value="15" />
            </div>
          </div>

          <div style="margin-top:10px">
            <label>Management style preference *</label>
            <div class="control">
              <label class="chip"><input type="radio" name="style" value="Passive" required /> Passive</label>
              <label class="chip"><input type="radio" name="style" value="Active" /> Active</label>
              <label class="chip"><input type="radio" name="style" value="Hybrid" /> Hybrid</label>
            </div>
          </div>
        </div>

        <!-- 7. Performance & Reporting -->
        <div class="section">
          <h2>7) Performance & Reporting</h2>
          <p class="hint">Benchmarks, report cadence, and transparency.</p>
          <div class="grid grid-2">
            <div>
              <label for="benchmark">Preferred benchmark</label>
              <select id="benchmark" name="benchmark">
                <option value="">Select…</option>
                <option>S&P 500</option>
                <option>MSCI World</option>
                <option>MSCI ACWI IMI (ESG)</option>
                <option>Custom (describe below)</option>
              </select>
            </div>
            <div>
              <label for="freq">Reporting frequency *</label>
              <select id="freq" name="freq" required>
                <option value="">Select…</option>
                <option>Monthly</option>
                <option>Quarterly</option>
                <option>Semi-annual</option>
                <option>Annual</option>
              </select>
            </div>
          </div>

          <div class="grid grid-2" style="margin-top:10px">
            <div>
              <label for="detail">Report detail level *</label>
              <select id="detail" name="detail" required>
                <option value="">Select…</option>
                <option>Summary</option>
                <option>Analytics (risk & attribution)</option>
                <option>Full (incl. transactions)</option>
              </select>
            </div>
            <div class="range-wrap">
              <label for="feeTransparency">Importance of cost transparency</label>
              <input id="feeTransparency" name="feeTransparency" type="range" min="0" max="10" value="8" />
              <div class="range-meta"><span>Low</span><span id="feeTransparencyVal">8/10</span><span>High</span></div>
            </div>
          </div>

          <div style="margin-top:10px">
            <label for="perfNotes">Benchmark or reporting notes</label>
            <textarea id="perfNotes" name="perfNotes" placeholder="Specify custom benchmark, data fields, or portals…"></textarea>
          </div>
        </div>

        <div class="btn-row">
          <button type="submit" class="btn-primary">Submit & Save</button>
        </div>
      </form>

      {% if saved %}
        <div class="section">
          <h2>Submission Saved</h2>
          <p class="hint">Saved to the database. A JSON copy is also available.</p>
          <p><a class="btn-primary" style="display:inline-block;text-decoration:none;padding:8px 12px;border-radius:10px"
                href="/download/{{ saved }}">Download JSON</a></p>
          {% if json_text %}
            <h3>Preview</h3>
            <pre class="json-out">{{ json_text }}</pre>
          {% endif %}
        </div>
      {% endif %}
    </div>
  </main>

  <script>
    // ---- Allocation bar ----
    const $ = (s, el=document) => el.querySelector(s);
    const allocInputs = ["allocEquities","allocFixed","allocAlts"].map(id=>$("#"+id));
    const allocBar = $("#allocBar"), allocTotal = $("#allocTotal");
    function updateAlloc(){
      if(!allocBar || !allocTotal) return;
      const sum = allocInputs.reduce((s,i)=> s + (parseFloat(i.value)||0), 0);
      allocTotal.textContent = (sum||0) + "%";
      allocBar.style.width = Math.max(0, Math.min(100, sum)) + "%";
      allocTotal.style.color = (sum===100) ? "var(--muted") : "#dc2626";
    }
    if (allocInputs.every(Boolean)) {
      allocInputs.forEach(i=> i.addEventListener("input", updateAlloc));
      updateAlloc();
    }

    // ---- Prevent Enter from submitting except in <textarea> ----
    (function(){
      const form = document.getElementById("qForm");
      if (!form) return;
      form.addEventListener("keydown", function(e){
        if (e.key === "Enter") {
          const tag = (e.target.tagName || "").toLowerCase();
          const type = (e.target.type || "").toLowerCase();
          const ok = (tag === "textarea") || (type === "submit");
          if (!ok) e.preventDefault();
        }
      });
    })();

    // ---- Local draft autosave/restore ----
    (function(){
      const form = document.getElementById("qForm");
      if (!form) return;
      const KEY = "questionnaire_draft_v1";

      try {
        const raw = localStorage.getItem(KEY);
        if (raw) {
          const draft = JSON.parse(raw);
          Object.entries(draft).forEach(([name, value]) => {
            const els = form.querySelectorAll(`[name="${CSS.escape(name)}"]`);
            if (!els.length) return;
            const first = els[0];
            if (first.type === "radio") {
              els.forEach(el => el.checked = (el.value === value));
            } else if (first.type === "checkbox") {
              const arr = Array.isArray(value) ? value : [value];
              els.forEach(el => el.checked = arr.includes(el.value));
            } else {
              first.value = value;
            }
          });
          updateAlloc();
          form.querySelectorAll('input[type="range"]').forEach(r => r.dispatchEvent(new Event('input', {bubbles:true})));
        }
      } catch(e){}

      function snapshot(){
        const data = {};
        const fd = new FormData(form);
        const multiKeys = new Set(["legal","esgExclude","esgFlags"]);
        for (const [k, v] of fd.entries()) {
          if (multiKeys.has(k)) {
            if (!Array.isArray(data[k])) data[k] = [];
            data[k].push(v);
          } else {
            data[k] = v;
          }
        }
        localStorage.setItem(KEY, JSON.stringify(data));
      }
      form.addEventListener("input", snapshot);
      form.addEventListener("change", snapshot);

      window.addEventListener("pageshow", function(){
        const successBanner = document.querySelector('.alert.success');
        if (successBanner) localStorage.removeItem(KEY);
      });
    })();
  </script>
</body>
</html>
"""

ADMIN_TPL = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Admin — Latest Submissions</title>
  <style>
    body{font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;background:#fff;color:#0b0f12;margin:0}
    .wrap{max-width:1000px;margin:0 auto;padding:20px}
    h1{margin:0 0 12px}
    table{border-collapse:collapse;width:100%}
    th,td{border:1px solid #e5e7eb;padding:8px 10px;text-align:left}
    th{background:#f8fafc}
    a{color:#2563eb;text-decoration:none} a:hover{text-decoration:underline}
    .meta{color:#6b7280;font-size:.9rem}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Latest Submissions</h1>
    <p class="meta"><a href="/">← Back to form</a></p>
    <table>
      <thead>
        <tr>
          <th>ID</th><th>When (UTC)</th><th>Name</th><th>Email</th>
          <th>Goal</th><th>Horizon</th><th>Style</th><th>Alloc (E/F/A)</th>
        </tr>
      </thead>
      <tbody>
        {% for r in rows %}
          <tr>
            <td>{{ r.id }}</td>
            <td>{{ r.created_at.strftime("%Y-%m-%d %H:%M:%S") }}</td>
            <td>{{ r.full_name }}</td>
            <td>{{ r.email }}</td>
            <td>{{ r.goal }}</td>
            <td>{{ r.horizon }}</td>
            <td>{{ r.style }}</td>
            <td>{{ "%.0f" % (r.alloc_equities or 0) }} /
                {{ "%.0f" % (r.alloc_fixed or 0) }} /
                {{ "%.0f" % (r.alloc_alts or 0) }}</td>
          </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)