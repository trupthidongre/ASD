"""
ASD Predictor — ASD Screening Flask Application
==============================================
RandomForest ML model (MINIPRO.ipynb) + full web platform
"""
import csv, io, os, secrets, uuid
from datetime import datetime, date, timedelta
from functools import wraps

import joblib, numpy as np, pandas as pd
from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, send_file, abort)
from werkzeug.security import generate_password_hash, check_password_hash

from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable, Image as RLImage)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ── App config ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "asdpredictor_dev_key_CHANGE_IN_PROD")

ADMIN_EMAIL    = os.environ.get("ADMIN_EMAIL",    "admin@gmail.com")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin@123")

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
USERS_FILE    = os.path.join(BASE_DIR, "users.csv")
ANSWERS_FILE  = os.path.join(BASE_DIR, "answers.csv")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.csv")
CHARTS_DIR    = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── ML Model ──────────────────────────────────────────────────────────────────
def _load_model():
    mp = os.path.join(BASE_DIR, "autism_model.joblib")
    ep = os.path.join(BASE_DIR, "autism_encoders.joblib")
    if not os.path.isfile(mp): mp = os.path.join(BASE_DIR, "autism_model.pkl")
    if not os.path.isfile(ep): ep = os.path.join(BASE_DIR, "autism_encoders.pkl")
    model = joblib.load(mp) if os.path.isfile(mp) else None
    encoders = joblib.load(ep) if os.path.isfile(ep) else {}
    if model: print(f"[ML] Model loaded: {os.path.basename(mp)}")
    else:      print("[ML] WARNING: Model not found. Run retrain_model.py first.")
    return model, encoders

ASD_MODEL, ASD_ENCODERS = _load_model()

MODEL_FEATURES = [
    "A1_Score","A2_Score","A3_Score","A4_Score","A5_Score",
    "A6_Score","A7_Score","A8_Score","A9_Score","A10_Score",
    "age","ethnicity","jundice","austim",
    "contry_of_res","used_app_before","result","relation"
]
CAT_COLS = ["ethnicity","contry_of_res","used_app_before","relation"]

# ── Age-group questions ───────────────────────────────────────────────────────
AGE_GROUP_QUESTIONS = {
    "child": [
        "Does your child avoid eye contact?",
        "Does your child have difficulty understanding feelings?",
        "Does your child repeat words or phrases?",
        "Does your child resist changes in routine?",
        "Does your child have unusual reactions to sounds or textures?",
        "Does your child play with toys in unusual ways?",
        "Does your child have difficulty making friends?",
        "Does your child show limited interest in social games?",
        "Does your child get upset easily over small changes?",
        "Does your child show intense interest in specific topics?"
    ],
    "teen": [
        "Do you find it hard to understand other people's emotions?",
        "Do you prefer routines over unexpected events?",
        "Do you struggle with social interactions?",
        "Do you repeat certain behaviors or phrases?",
        "Do you get anxious in social situations?",
        "Do you have intense focus on special interests?",
        "Do you avoid eye contact?",
        "Do you find it hard to make friends?",
        "Do you find sensory experiences overwhelming?",
        "Do you feel isolated from peers?"
    ],
    "adult": [
        "Do you struggle with understanding social cues?",
        "Do you prefer routines and schedules?",
        "Do you have difficulty maintaining relationships?",
        "Do you focus intensely on specific interests?",
        "Do you avoid eye contact in conversations?",
        "Do you feel anxious in social gatherings?",
        "Do you struggle to interpret others' emotions?",
        "Do you repeat certain phrases or habits?",
        "Do you feel overwhelmed in sensory-rich environments?",
        "Do you find small talk challenging?"
    ],
    "middle": [
        "Do you have difficulty reading social cues?",
        "Do you prefer predictable routines?",
        "Do you have intense focus on hobbies or topics?",
        "Do you find group interactions stressful?",
        "Do you avoid eye contact?",
        "Do you struggle with communication nuances?",
        "Do you repeat certain behaviors or actions?",
        "Do you feel anxious in crowded places?",
        "Do you prefer solitary activities?",
        "Do you find it hard to maintain friendships?"
    ],
    "senior": [
        "Do you avoid social interactions?",
        "Do you have difficulty understanding emotions of others?",
        "Do you prefer strict routines?",
        "Do you repeat certain activities or phrases?",
        "Do you feel overwhelmed by noise or crowds?",
        "Do you have intense focus on particular interests?",
        "Do you avoid eye contact?",
        "Do you struggle with communication?",
        "Do you prefer solitary hobbies?",
        "Do you find social situations exhausting?"
    ]
}

# ── CSRF ──────────────────────────────────────────────────────────────────────
def generate_csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(24)
    return session["csrf_token"]

def validate_csrf():
    t = session.get("csrf_token","")
    if not t or t != request.form.get("csrf_token",""):
        abort(403)

app.jinja_env.globals["csrf_token"] = generate_csrf_token

# ── Auth decorators ───────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def dec(*a, **kw):
        if "user_email" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*a, **kw)
    return dec

def admin_required(f):
    @wraps(f)
    def dec(*a, **kw):
        if not session.get("is_admin"):
            flash("Admin login required.", "danger")
            return redirect(url_for("admin_login"))
        return f(*a, **kw)
    return dec

# ── CSV helpers ───────────────────────────────────────────────────────────────
def read_csv(path):
    if not os.path.isfile(path): return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def append_csv(path, fields, row):
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists: w.writeheader()
        w.writerow(row)

# ── ML helpers ────────────────────────────────────────────────────────────────
def _safe_encode(enc, val):
    s = str(val).strip()
    if s in enc.classes_: return int(enc.transform([s])[0])
    lo = s.lower()
    for c in enc.classes_:
        if c.lower() == lo: return int(enc.transform([c])[0])
    return 0

def _age_from_dob(dob_str):
    try:
        d = datetime.strptime(dob_str, "%Y-%m-%d").date()
        t = date.today()
        return t.year - d.year - ((t.month, t.day) < (d.month, d.day))
    except: return 25

def predict_asd(answers, extra, sess):
    if ASD_MODEL is None:
        raise RuntimeError("autism_model.joblib not found. Run retrain_model.py first.")

    total = sum(answers)
    row = {
        "A1_Score":answers[0],"A2_Score":answers[1],"A3_Score":answers[2],
        "A4_Score":answers[3],"A5_Score":answers[4],"A6_Score":answers[5],
        "A7_Score":answers[6],"A8_Score":answers[7],"A9_Score":answers[8],
        "A10_Score":answers[9],
        "age": _age_from_dob(sess.get("user_dob","")),
        "result": total,
        "jundice": 1 if extra.get("jundice","no").lower()=="yes" else 0,
        "austim":  1 if extra.get("family_asd","no").lower()=="yes" else 0,
        "ethnicity":       extra.get("ethnicity","Others"),
        "contry_of_res":   extra.get("contry_of_res","United States"),
        "used_app_before": extra.get("used_app_before","no"),
        "relation":        extra.get("relation","Self"),
    }
    df = pd.DataFrame([row], columns=MODEL_FEATURES)
    for col in CAT_COLS:
        df[col] = _safe_encode(ASD_ENCODERS[col], df.at[0,col]) if col in ASD_ENCODERS else 0

    pred     = int(ASD_MODEL.predict(df)[0])
    proba    = ASD_MODEL.predict_proba(df)[0]
    prob_asd = round(float(proba[1])*100, 1)
    prob_no  = round(float(proba[0])*100, 1)

    if pred == 0:
        rl, bc, pct = "Low Risk", "success", prob_no
        rec = ("The RandomForest ML model predicts a low likelihood of ASD. "
               "Continue monitoring developmental milestones and maintain "
               "regular health check-ups. Consult a GP if specific concerns arise.")
    elif prob_asd < 70:
        rl, bc, pct = "Moderate Risk", "warning", prob_asd
        rec = (f"The RandomForest ML model has detected moderate ASD-associated "
               f"indicators (confidence: {prob_asd}%). We recommend a formal "
               f"evaluation with a certified developmental paediatrician or "
               f"clinical psychologist.")
    else:
        rl, bc, pct = "High Risk", "danger", prob_asd
        rec = (f"The RandomForest ML model predicts a high likelihood of ASD "
               f"(confidence: {prob_asd}%). We strongly recommend an urgent "
               f"consultation with a licensed specialist for a formal diagnostic evaluation.")

    return {"prediction":pred,"probability":prob_asd,"prob_no_asd":prob_no,
            "risk_level":rl,"badge_class":bc,"percentage":pct,
            "recommendation":rec,"total_score":total,"ml_used":True}

# ── Charts ────────────────────────────────────────────────────────────────────
def make_charts(answers, chart_id):
    yes_c = sum(answers); no_c = 10 - yes_c
    pie_f = f"pie_{chart_id}.png"; bar_f = f"bar_{chart_id}.png"
    pie_p = os.path.join(CHARTS_DIR, pie_f); bar_p = os.path.join(CHARTS_DIR, bar_f)

    fig, ax = plt.subplots(figsize=(5,4))
    ax.pie([yes_c,no_c], labels=[f"Yes ({yes_c})",f"No ({no_c})"],
           autopct="%1.1f%%", colors=["#e74c3c","#2ecc71"],
           explode=(0.05,0.05), startangle=90,
           wedgeprops={"edgecolor":"white","linewidth":2.5})
    ax.set_title("Response Distribution", fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("#f8f9fa"); plt.tight_layout()
    plt.savefig(pie_p, dpi=120, bbox_inches="tight", facecolor="#f8f9fa"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5,3.5))
    bars = ax.bar(["Your Score","Max Score"],[yes_c,10],
                  color=["#6a1b9a","#bdc3c7"], edgecolor="white", linewidth=1.5, width=0.45)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.15,
                str(int(b.get_height())), ha="center", va="bottom",
                fontweight="bold", fontsize=12)
    ax.set_ylim(0,12); ax.set_ylabel("Score")
    ax.set_title("Score vs Maximum (10)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#f8f9fa")
    plt.tight_layout()
    plt.savefig(bar_p, dpi=120, bbox_inches="tight", facecolor="#f8f9fa"); plt.close(fig)

    return f"charts/{pie_f}", f"charts/{bar_f}"

def make_admin_charts(risk_counts, trend_data):
    rb_p = os.path.join(CHARTS_DIR,"admin_risk_bar.png")
    tl_p = os.path.join(CHARTS_DIR,"admin_trend_line.png")

    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(list(risk_counts.keys()), list(risk_counts.values()),
                  color=["#2ecc71","#f39c12","#e74c3c"], edgecolor="white",
                  linewidth=1.5, width=0.5)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
                str(int(b.get_height())), ha="center", va="bottom",
                fontweight="bold", fontsize=12)
    ax.set_title("Risk Level Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Assessments")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#f8f9fa")
    plt.tight_layout(); plt.savefig(rb_p, dpi=120, bbox_inches="tight", facecolor="#f8f9fa"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4))
    if trend_data:
        dates  = [d[0] for d in trend_data]; counts = [d[1] for d in trend_data]
        ax.plot(dates, counts, marker="o", color="#6a1b9a", linewidth=2.5,
                markersize=7, markerfacecolor="white", markeredgewidth=2.5)
        ax.fill_between(range(len(dates)), counts, alpha=0.15, color="#9b59b6")
        ax.set_xticks(range(len(dates))); ax.set_xticklabels(dates, rotation=30, ha="right", fontsize=8)
        for i,(x,y) in enumerate(zip(range(len(dates)),counts)):
            ax.annotate(str(y),(x,y),textcoords="offset points",xytext=(0,8),ha="center",fontsize=9,fontweight="bold")
    else:
        ax.text(0.5,0.5,"No data yet",ha="center",va="center",transform=ax.transAxes,fontsize=14,color="gray")
    ax.set_title("Assessments Over Time",fontsize=14,fontweight="bold")
    ax.set_ylabel("Per Day")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#f8f9fa")
    plt.tight_layout(); plt.savefig(tl_p, dpi=120, bbox_inches="tight", facecolor="#f8f9fa"); plt.close(fig)

    return "charts/admin_risk_bar.png","charts/admin_trend_line.png"

# ── PDF ───────────────────────────────────────────────────────────────────────
def build_pdf(r):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2.2*cm, rightMargin=2.2*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm,
                            title="ASD Screening Report")
    styles = getSampleStyleSheet()
    title_s  = ParagraphStyle("T",fontSize=22,textColor=colors.HexColor("#1a1a2e"),
                               spaceAfter=4,alignment=TA_CENTER,fontName="Helvetica-Bold")
    sub_s    = ParagraphStyle("S",fontSize=10,textColor=colors.HexColor("#6c757d"),
                               spaceAfter=2,alignment=TA_CENTER,fontName="Helvetica")
    head_s   = ParagraphStyle("H",fontSize=13,textColor=colors.HexColor("#1a1a2e"),
                               spaceBefore=14,spaceAfter=6,fontName="Helvetica-Bold")
    body_s   = ParagraphStyle("B",fontSize=10,leading=16,textColor=colors.HexColor("#333333"),
                               spaceAfter=4,fontName="Helvetica")
    footer_s = ParagraphStyle("F",fontSize=8,textColor=colors.HexColor("#999999"),
                               alignment=TA_CENTER,fontName="Helvetica")

    RISK_COLOR = {"Low Risk":colors.HexColor("#28a745"),
                  "Moderate Risk":colors.HexColor("#ffc107"),
                  "High Risk":colors.HexColor("#dc3545")}
    rc = RISK_COLOR.get(r.get("risk_level",""), colors.HexColor("#333333"))

    story = []
    story.append(Paragraph("ASD Predictor", title_s))
    story.append(Paragraph("ASD Clinical Screening Report", sub_s))
    story.append(Paragraph(f"Generated: {r.get('timestamp','')}", sub_s))
    story.append(Spacer(1,0.5*cm))
    story.append(HRFlowable(width="100%",thickness=2,color=colors.HexColor("#0a8f8f"),spaceBefore=4,spaceAfter=12))

    story.append(Paragraph("Patient Information", head_s))
    pt_data = [
        ["Full Name",    r.get("name","—")],
        ["Date of Birth",r.get("dob","—")],
        ["Gender",       r.get("gender","—")],
        ["Age Group",    r.get("age_group","—").title()],
        ["Email",        r.get("email","—")],
    ]
    pt_table = Table(pt_data, colWidths=[5*cm,11*cm])
    pt_table.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1),10),
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
        ("TEXTCOLOR",(0,0),(0,-1),colors.HexColor("#555555")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#f8f9fa"),colors.white]),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#dee2e6")),
        ("PADDING",(0,0),(-1,-1),7),
    ]))
    story.append(pt_table)
    story.append(Spacer(1,0.5*cm))

    story.append(Paragraph("Assessment Results", head_s))
    score_data = [
        ["Metric","Value"],
        ["ASD Score", f"{r.get('total_score',0)} / 10"],
        ["ASD Probability", f"{r.get('probability',r.get('percentage',0))}%"],
        ["No-ASD Probability", f"{r.get('prob_no_asd',0)}%"],
        ["Risk Level", r.get("risk_level","—")],
        ["Model Used", "RandomForest ML" if r.get("ml_used") else "Score-based"],
    ]
    sc_table = Table(score_data, colWidths=[7*cm,9*cm])
    sc_table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0a8f8f")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1),10),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f8f9fa"),colors.white]),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#dee2e6")),
        ("PADDING",(0,0),(-1,-1),8),
        ("TEXTCOLOR",(1,4),(1,4),rc),
        ("FONTNAME",(1,4),(1,4),"Helvetica-Bold"),
    ]))
    story.append(sc_table)
    story.append(Spacer(1,0.5*cm))

    # Pie chart
    pie_abs = os.path.join(CHARTS_DIR, f"pie_{r.get('chart_id','')}.png")
    if os.path.isfile(pie_abs):
        story.append(Paragraph("Response Visualisation", head_s))
        story.append(RLImage(pie_abs, width=10*cm, height=8*cm))
        story.append(Spacer(1,0.3*cm))

    story.append(Paragraph("Clinical Recommendation", head_s))
    rec_table = Table([[Paragraph(r.get("recommendation",""), body_s)]], colWidths=[16*cm])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#f8f9fa")),
        ("BOX",(0,0),(-1,-1),1,rc),
        ("PADDING",(0,0),(-1,-1),12),
        ("ROUNDEDCORNERS",[6]),
    ]))
    story.append(rec_table)
    story.append(Spacer(1,0.8*cm))
    story.append(HRFlowable(width="100%",thickness=1,color=colors.HexColor("#dee2e6"),spaceBefore=12))
    story.append(Paragraph(
        "ASD Predictor — Automated Screening System  |  "
        f"Report generated on {r.get('timestamp','')}",
        footer_s))
    story.append(Paragraph(
        "⚠️ This document is for informational purposes only and does not constitute a clinical diagnosis.",
        footer_s))

    doc.build(story)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — PUBLIC
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        validate_csrf()
        name  = request.form.get("name","").strip()
        dob   = request.form.get("dob","").strip()
        gender= request.form.get("gender","").strip()
        email = request.form.get("email","").strip().lower()
        phone = request.form.get("phone","").strip()
        pw    = request.form.get("password","").strip()
        cpw   = request.form.get("confirm_password","").strip()

        _VALID = {"child","teen","adult","middle","senior"}
        ag = request.form.get("age_group","").strip().lower()
        if ag not in _VALID:
            ag = "adult"
            try:
                d = datetime.strptime(dob,"%Y-%m-%d").date(); av = date.today().year-d.year-((date.today().month,date.today().day)<(d.month,d.day))
                ag = "child" if av<12 else "teen" if av<18 else "adult" if av<40 else "middle" if av<60 else "senior"
            except: pass

        if pw != cpw:
            return render_template("register.html", msg="Passwords do not match!", today=date.today().isoformat())
        existing = read_csv(USERS_FILE)
        if any(r.get("email","").lower()==email for r in existing):
            return render_template("register.html", msg="Email already registered!", today=date.today().isoformat())
        append_csv(USERS_FILE, ["name","dob","age_group","gender","email","phone","password"], {
            "name":name,"dob":dob,"age_group":ag,"gender":gender,
            "email":email,"phone":phone,"password":generate_password_hash(pw)
        })
        flash("Registration successful! Please sign in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html", today=date.today().isoformat())

@app.route("/login", methods=["GET","POST"])
def login():
    if "user_email" in session: return redirect(url_for("questions"))
    if request.method == "POST":
        validate_csrf()
        email = request.form.get("email","").strip().lower()
        pw    = request.form.get("password","").strip()
        users = read_csv(USERS_FILE)
        user  = next((u for u in users if u.get("email","").lower()==email), None)
        if user and check_password_hash(user["password"], pw):
            session["user_email"]    = email
            session["user_name"]     = user.get("name","")
            ag = user.get("age_group","adult").strip().lower()
            session["user_age_group"]= ag if ag in AGE_GROUP_QUESTIONS else "adult"
            session["user_dob"]      = user.get("dob","")
            session["user_gender"]   = user.get("gender","")
            session["chart_id"]      = str(uuid.uuid4())[:8]
            return redirect(url_for("questions"))
        return render_template("login.html", msg="Invalid email or password.")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/questions", methods=["GET","POST"])
@login_required
def questions():
    ag     = session.get("user_age_group","adult")
    ag     = ag if ag in AGE_GROUP_QUESTIONS else "adult"
    q_list = AGE_GROUP_QUESTIONS[ag]

    if request.method == "POST":
        validate_csrf()
        answers = [1 if request.form.get(f"Q{i}","no").strip().lower()=="yes" else 0 for i in range(1,11)]
        rj = request.form.get("jundice","0"); rf = request.form.get("family_asd","0")
        extra = {
            "ethnicity":       request.form.get("ethnicity","Others").strip(),
            "jundice":         "yes" if rj in ("1","yes") else "no",
            "family_asd":      "yes" if rf in ("1","yes") else "no",
            "contry_of_res":   request.form.get("contry_of_res","United States").strip(),
            "used_app_before": request.form.get("used_app_before","no").strip().lower(),
            "relation":        request.form.get("relation","Self").strip(),
            "additional_info": request.form.get("additional_info","").strip(),
        }
        chart_id = session.get("chart_id", str(uuid.uuid4())[:8])
        try:
            pred = predict_asd(answers, extra, session)
        except RuntimeError as e:
            flash(str(e), "danger"); return redirect(url_for("questions"))
        except Exception as e:
            flash(f"Prediction error: {e}", "danger"); return redirect(url_for("questions"))

        pie, bar = make_charts(answers, chart_id)
        result = {
            **pred,
            "name":      session.get("user_name",""),
            "email":     session.get("user_email",""),
            "dob":       session.get("user_dob",""),
            "gender":    session.get("user_gender",""),
            "age_group": ag,
            "answers":   ["yes" if a==1 else "no" for a in answers],
            "pie_chart": pie, "bar_chart": bar,
            "chart_id":  chart_id,
            "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
            "ethnicity":       extra["ethnicity"],
            "jundice":         extra["jundice"],
            "family_asd":      extra["family_asd"],
            "contry_of_res":   extra["contry_of_res"],
            "used_app_before": extra["used_app_before"],
            "relation":        extra["relation"],
        }
        session["result"] = result
        # Save to answers.csv
        fields = (["email","name","age_group"] +
                  [f"Q{i}" for i in range(1,11)] +
                  ["total_score","risk_level","prediction","probability",
                   "ethnicity","jundice","family_asd","contry_of_res",
                   "used_app_before","relation","ml_used","timestamp"])
        append_csv(ANSWERS_FILE, fields, {
            "email": session.get("user_email",""), "name": session.get("user_name",""),
            "age_group": ag,
            **{f"Q{i+1}": answers[i] for i in range(10)},
            "total_score": pred["total_score"], "risk_level": pred["risk_level"],
            "prediction": pred["prediction"], "probability": pred["probability"],
            "ethnicity": extra["ethnicity"], "jundice": extra["jundice"],
            "family_asd": extra["family_asd"], "contry_of_res": extra["contry_of_res"],
            "used_app_before": extra["used_app_before"], "relation": extra["relation"],
            "ml_used": pred["ml_used"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        return redirect(url_for("result"))
    return render_template("questions.html", age_group=ag, questions=q_list)

@app.route("/result")
@login_required
def result():
    r = session.get("result")
    if not r:
        flash("No assessment data. Please complete the questionnaire.", "warning")
        return redirect(url_for("questions"))
    return render_template("result.html", result=r)

@app.route("/download_report")
@login_required
def download_report():
    r = session.get("result")
    if not r:
        flash("No assessment data.", "warning")
        return redirect(url_for("questions"))
    buf  = build_pdf(r)
    name = r.get("name","patient").replace(" ","_")
    return send_file(buf, as_attachment=True,
                     download_name=f"ASD_Report_{name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                     mimetype="application/pdf")

@app.route("/feedback", methods=["GET","POST"])
@login_required
def feedback():
    if request.method == "POST":
        validate_csrf()
        append_csv(FEEDBACK_FILE, ["email","rating","category","message","timestamp"], {
            "email":    session.get("user_email",""),
            "rating":   request.form.get("rating","").strip(),
            "category": request.form.get("category","").strip(),
            "message":  request.form.get("message","").strip(),
            "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        return redirect(url_for("thankyou"))
    return render_template("feedback.html")

@app.route("/thankyou")
@login_required
def thankyou():
    return render_template("thankyou.html")

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — ADMIN
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/admin/login", methods=["GET","POST"])
def admin_login():
    if session.get("is_admin"): return redirect(url_for("admin_dashboard"))
    if request.method == "POST":
        validate_csrf()
        un = request.form.get("username","").strip()
        pw = request.form.get("password","").strip()
        if un.lower()==ADMIN_EMAIL.lower() and pw==ADMIN_PASSWORD:
            session["is_admin"] = True
            return redirect(url_for("admin_dashboard"))
        flash("Invalid admin credentials.", "danger")
    return render_template("admin_login.html")

@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin",None)
    flash("Admin logged out.", "info")
    return redirect(url_for("admin_login"))

@app.route("/admin")
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    users     = read_csv(USERS_FILE)
    answers   = read_csv(ANSWERS_FILE)
    feedbacks = read_csv(FEEDBACK_FILE)

    scores      = []
    risk_counts = {"Low Risk":0,"Moderate Risk":0,"High Risk":0}
    for row in answers:
        try: scores.append(int(row.get("total_score",0)))
        except: pass
        rl = row.get("risk_level","")
        if rl in risk_counts: risk_counts[rl] += 1

    avg_score = round(sum(scores)/len(scores),2) if scores else 0
    daily     = {}
    for row in answers:
        ts = row.get("timestamp","")
        if ts: daily[ts[:10]] = daily.get(ts[:10],0)+1
    trend = sorted(daily.items())[-14:]
    rc, tc = make_admin_charts(risk_counts, trend)

    stats = {"total_users":len(users),"total_assessments":len(answers),
             "avg_score":avg_score,"risk_counts":risk_counts,
             "risk_chart":rc,"trend_chart":tc,"total_feedback":len(feedbacks)}
    return render_template("admin.html", stats=stats, answers=answers[-10:])

# ══════════════════════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════════════════
@app.errorhandler(403)
def forbidden(e):
    return render_template("error.html", error_code=403, error_title="Access Denied",
                           error_message="CSRF token mismatch or insufficient permissions."), 403
@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", error_code=404, error_title="Page Not Found",
                           error_message="The page you're looking for doesn't exist."), 404
@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", error_code=500, error_title="Server Error",
                           error_message="Internal server error. Please try again later."), 500

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG","0")=="1"
    port  = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port, debug=debug)
