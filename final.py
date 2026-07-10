import os
import json
import sqlite3
import secrets
import html
import joblib
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd
from flask import Flask, request, redirect, url_for, session, flash, render_template_string, g, make_response, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

try:
    import joblib
except Exception:
    joblib = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
except Exception:
    A4 = None
    colors = None
    getSampleStyleSheet = None
    ParagraphStyle = None
    TA_LEFT = None
    mm = None
    SimpleDocTemplate = None
    Paragraph = None
    Spacer = None
    Table = None
    TableStyle = None

# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "SQA Triple Portal"
DB_PATH = "sqa_dual_portal.db"
MODEL_PATH = "sqa_supplier_score_regressor_v4_safe_pca.joblib"
METRICS_PATH = "sqa_training_metrics_v4_safe_pca.json"
OLDER_DATA_CSV = "testing.csv"

SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(24))
PORT = int(os.environ.get("PORT", "5000"))
DEBUG = True
WATCHLIST_BAND = 0.05

ROLE_QUALITY = "quality_tester"
ROLE_DEALER = "dealer"
ROLE_COMPANY = "company_user"
ROLE_AUDITOR = "auditor"
ALLOWED_ROLES = [ROLE_QUALITY, ROLE_DEALER, ROLE_COMPANY,ROLE_AUDITOR]

PART_GROUPS = {
    "Mechanical / Wear-Driven": ["V-Belt", "Wheel Bearing", "U-Joint"],
    "Fluid / Pressure-Driven": ["Radiator Hose", "Brake Chamber", "Relay Valve", "Fuel/Water Separator"],
    "Electrical / Signal-Driven": ["Air Filter", "Wheel Speed Sensor"],
}

PART_TO_CATEGORY = {
    "V-Belt": "Mechanical / Wear-Driven",
    "Wheel Bearing": "Mechanical / Wear-Driven",
    "U-Joint": "Mechanical / Wear-Driven",
    "Radiator Hose": "Fluid / Pressure-Driven",
    "Brake Chamber": "Fluid / Pressure-Driven",
    "Relay Valve": "Fluid / Pressure-Driven",
    "Fuel/Water Separator": "Fluid / Pressure-Driven",
    "Air Filter": "Electrical / Signal-Driven",
    "Wheel Speed Sensor": "Electrical / Signal-Driven",
}

CORE_COLUMNS = [
    "test_date", "plant_name", "vehicle_model", "supplier_id", "supplier_name",
    "part_name", "part_category", "importance_band", "data_source_type",
    "market_price_inr", "qty_inspected", "qty_defective", "ppm", "otd_pct",
    "audit_score_pct", "cpk", "criticality_weight_0_1", "min_required_sqm_0_1",
    "impact_raw"
]

NUMERIC_COLUMNS = {
    "market_price_inr", "qty_inspected", "qty_defective", "ppm", "otd_pct",
    "audit_score_pct", "cpk", "criticality_weight_0_1", "min_required_sqm_0_1",
    "impact_raw"
}

FEATURE_COLUMNS = []
MODEL = None
MODEL_INFO = {}

PART_SPEC_CONFIG = {
    "V-Belt": {
        "title": "V-Belt / Alternator Belt Specific Metrics",
        "help": "Enter the belt-condition signals used to compute the part metric.",
        "fields": [
            {"name": "belt_tension_dev_ratio", "label": "Belt Tension Deviation Ratio", "type": "number", "step": "0.01", "min": "0", "max": "1", "placeholder": "0.08"},
            {"name": "slip_ratio", "label": "Slip Ratio", "type": "number", "step": "0.01", "min": "0", "max": "1", "placeholder": "0.03"},
            {"name": "crack_density", "label": "Crack Density (per 100 mm)", "type": "number", "step": "0.01", "min": "0", "placeholder": "2"},
            {"name": "crack_density_max", "label": "Max Allowed Crack Density", "type": "number", "step": "0.01", "min": "0", "placeholder": "10", "default": "10"},
        ],
    },
    "Radiator Hose": {
        "title": "Radiator Hose Specific Metrics",
        "help": "Enter cooling-hose pressure and leak stability inputs.",
        "fields": [
            {"name": "pressure_retention_ratio", "label": "Pressure Retention Ratio", "type": "number", "step": "0.01", "min": "0", "max": "1", "placeholder": "0.96"},
            {"name": "leak_rate", "label": "Leak Rate", "type": "number", "step": "0.01", "min": "0", "placeholder": "0.20"},
            {"name": "leak_rate_max", "label": "Max Leak Rate", "type": "number", "step": "0.01", "min": "0", "placeholder": "2.0", "default": "2.0"},
            {"name": "temp_stability_dev", "label": "Temperature Stability Deviation", "type": "number", "step": "0.01", "min": "0", "max": "1", "placeholder": "0.05"},
        ],
    },
    "Air Filter": {
        "title": "Air Filter Specific Metrics",
        "help": "Enter restriction, efficiency, and fuel-penalty related signals.",
        "fields": [
            {"name": "restriction_dp", "label": "Restriction ΔP", "type": "number", "step": "0.01", "min": "0", "placeholder": "12"},
            {"name": "restriction_dp_max", "label": "Max Restriction ΔP", "type": "number", "step": "0.01", "min": "0", "placeholder": "30", "default": "30"},
            {"name": "filtration_efficiency", "label": "Filtration Efficiency", "type": "number", "step": "0.01", "min": "0", "max": "1", "placeholder": "0.98"},
            {"name": "efficiency_min", "label": "Minimum Efficiency", "type": "number", "step": "0.01", "min": "0", "max": "1", "placeholder": "0.95", "default": "0.95"},
            {"name": "fuel_penalty_ratio", "label": "Fuel Penalty Ratio", "type": "number", "step": "0.01", "min": "0", "max": "1", "placeholder": "0.02"},
            {"name": "fuel_penalty_max", "label": "Max Fuel Penalty Ratio", "type": "number", "step": "0.01", "min": "0", "max": "1", "placeholder": "0.08", "default": "0.08"},
        ],
    },
    "Brake Chamber": {
        "title": "Brake Chamber Specific Metrics",
        "help": "Enter safety-critical airtightness and stroke data.",
        "fields": [
            {"name": "brake_leak_rate", "label": "Brake Leak Rate", "type": "number", "step": "0.01", "min": "0", "placeholder": "0.10"},
            {"name": "brake_leak_rate_max", "label": "Max Brake Leak Rate", "type": "number", "step": "0.01", "min": "0", "placeholder": "1.0", "default": "1.0"},
            {"name": "pushrod_stroke", "label": "Applied Pushrod Stroke", "type": "number", "step": "0.01", "min": "0", "placeholder": "18"},
            {"name": "stroke_limit", "label": "Stroke Limit", "type": "number", "step": "0.01", "min": "0", "placeholder": "30", "default": "30"},
            {"name": "response_lag", "label": "Response Lag", "type": "number", "step": "0.01", "min": "0", "placeholder": "0.12"},
            {"name": "response_lag_max", "label": "Max Response Lag", "type": "number", "step": "0.01", "min": "0", "placeholder": "0.5", "default": "0.5"},
        ],
    },
    "Relay Valve": {
        "title": "Relay Valve Specific Metrics",
        "help": "Enter lag, pressure delivery and leakage values.",
        "fields": [
            {"name": "relay_lag", "label": "Relay Lag", "type": "number", "step": "0.01", "min": "0", "placeholder": "0.08"},
            {"name": "relay_lag_max", "label": "Max Relay Lag", "type": "number", "step": "0.01", "min": "0", "placeholder": "0.4", "default": "0.4"},
            {"name": "pressure_delivery_ratio", "label": "Pressure Delivery Ratio", "type": "number", "step": "0.01", "min": "0", "max": "2", "placeholder": "0.98"},
            {"name": "pressure_target_ratio", "label": "Pressure Target Ratio", "type": "number", "step": "0.01", "min": "0", "max": "2", "placeholder": "1.0", "default": "1.0"},
            {"name": "relay_leak_rate", "label": "Relay Leak Rate", "type": "number", "step": "0.01", "min": "0", "placeholder": "0.10"},
            {"name": "relay_leak_rate_max", "label": "Max Relay Leak Rate", "type": "number", "step": "0.01", "min": "0", "placeholder": "1.0", "default": "1.0"},
        ],
    },
    "Wheel Speed Sensor": {
        "title": "Wheel Speed Sensor Specific Metrics",
        "help": "Enter pulse integrity and motion-agreement inputs.",
        "fields": [
            {"name": "pulse_integrity", "label": "Pulse Integrity", "type": "number", "step": "0.01", "min": "0", "max": "1", "placeholder": "0.99"},
            {"name": "speed_dev_ratio", "label": "Speed Deviation Ratio", "type": "number", "step": "0.01", "min": "0", "max": "1", "placeholder": "0.03"},
            {"name": "speed_dev_max", "label": "Max Speed Deviation Ratio", "type": "number", "step": "0.01", "min": "0", "max": "1", "placeholder": "0.10", "default": "0.10"},
            {"name": "dropout_count", "label": "Dropout Count", "type": "number", "step": "1", "min": "0", "placeholder": "0"},
            {"name": "dropout_max", "label": "Max Dropout Count", "type": "number", "step": "1", "min": "1", "placeholder": "5", "default": "5"},
        ],
    },
    "Wheel Bearing": {
        "title": "Wheel Bearing Specific Metrics",
        "help": "Enter temperature, vibration, and play values.",
        "fields": [
            {"name": "bearing_temp_rise", "label": "Bearing Temperature Rise", "type": "number", "step": "0.01", "min": "0", "placeholder": "10"},
            {"name": "bearing_temp_max", "label": "Max Bearing Temperature Rise", "type": "number", "step": "0.01", "min": "0", "placeholder": "45", "default": "45"},
            {"name": "bearing_vibration_rms", "label": "Bearing Vibration RMS", "type": "number", "step": "0.01", "min": "0", "placeholder": "1.5"},
            {"name": "bearing_vibration_max", "label": "Max Bearing Vibration RMS", "type": "number", "step": "0.01", "min": "0", "placeholder": "4.0", "default": "4.0"},
            {"name": "bearing_play", "label": "Bearing Play", "type": "number", "step": "0.01", "min": "0", "placeholder": "0.08"},
            {"name": "bearing_play_max", "label": "Max Bearing Play", "type": "number", "step": "0.01", "min": "0", "placeholder": "0.30", "default": "0.30"},
        ],
    },
    "U-Joint": {
        "title": "U-Joint Specific Metrics",
        "help": "Enter driveline vibration, backlash, and temperature inputs.",
        "fields": [
            {"name": "ujoint_vibration_rms", "label": "U-Joint Vibration RMS", "type": "number", "step": "0.01", "min": "0", "placeholder": "1.8"},
            {"name": "ujoint_vibration_max", "label": "Max U-Joint Vibration RMS", "type": "number", "step": "0.01", "min": "0", "placeholder": "5.0", "default": "5.0"},
            {"name": "ujoint_backlash", "label": "U-Joint Backlash", "type": "number", "step": "0.01", "min": "0", "placeholder": "0.12"},
            {"name": "ujoint_backlash_max", "label": "Max U-Joint Backlash", "type": "number", "step": "0.01", "min": "0", "placeholder": "0.50", "default": "0.50"},
            {"name": "ujoint_temp_rise", "label": "U-Joint Temperature Rise", "type": "number", "step": "0.01", "min": "0", "placeholder": "6"},
            {"name": "ujoint_temp_max", "label": "Max U-Joint Temperature Rise", "type": "number", "step": "0.01", "min": "0", "placeholder": "40", "default": "40"},
        ],
    },
    "Fuel/Water Separator": {
        "title": "Fuel / Water Separator Specific Metrics",
        "help": "Enter contamination, differential pressure and maintenance compliance values.",
        "fields": [
            {"name": "water_out_ppm", "label": "Water Contamination Downstream (ppm)", "type": "number", "step": "0.01", "min": "0", "placeholder": "15"},
            {"name": "water_out_max", "label": "Max Water Contamination (ppm)", "type": "number", "step": "0.01", "min": "0", "placeholder": "100", "default": "100"},
            {"name": "separator_dp", "label": "Separator Differential Pressure", "type": "number", "step": "0.01", "min": "0", "placeholder": "8"},
            {"name": "separator_dp_max", "label": "Max Separator Differential Pressure", "type": "number", "step": "0.01", "min": "0", "placeholder": "25", "default": "25"},
            {"name": "service_compliance_ratio", "label": "Service Compliance Ratio", "type": "number", "step": "0.01", "min": "0", "max": "1", "placeholder": "1.0"},
        ],
    },
}

REPORT_TITLE = "SQA Verification Report"

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_PERMANENT"] = False


# ============================================================
# UTILITY HELPERS
# ============================================================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def pct_or_zero(value):
    try:
        if value is None:
            return 0.0
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return 0.0
        return round(float(value), 4)
    except Exception:
        return 0.0


def clamp(value, low=0.0, high=1.0):
    try:
        return max(low, min(float(value), high))
    except Exception:
        return low


def find_part_category(part_name):
    return PART_TO_CATEGORY.get((part_name or "").strip(), "")


def role_label(role):
    return {
        ROLE_QUALITY: "Quality Tester",
        ROLE_DEALER: "Dealer",
        ROLE_COMPANY: "Company Manufacturing User",
        ROLE_AUDITOR: "Auditor",
    }.get(role, role.replace("_", " ").title())

def load_auditor_data():

    data = {}

    try:
        data["sensor"] = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "sensor_anomaly_results.csv"
    )
)
    except:
        data["sensor"] = pd.DataFrame()

    try:
        data["supplier"] = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "supplier_predictions.csv"
    )
)
    except:
        data["supplier"] = pd.DataFrame()

    try:
        data["fraud"] = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "claim_fraud_predictions.csv"
    )
)
    except:
        data["fraud"] = pd.DataFrame()

    try:
        data["rca"] = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "unified_rca_results.csv"
    )
)

    except:
        data["rca"] = pd.DataFrame()

    return data

def generate_dashboard_charts():

    static_dir = os.path.join(
        BASE_DIR,
        "static"
    )

    os.makedirs(
        static_dir,
        exist_ok=True
    )

    data = load_auditor_data()

    supplier_df = data["supplier"]
    fraud_df = data["fraud"]
    sensor_df = data["sensor"]
    rca_df = data["rca"]

    if supplier_df.empty:
        return
def generate_record_code(record_id):
    return f"SQA-{10000 + record_id}"

    # ====================================================
    # Supplier Histogram
    # ====================================================

    if (
        not supplier_df.empty
        and "predicted_supplier_score"
        in supplier_df.columns
    ):

        plt.figure(
            figsize=(10, 5)
        )

        plt.hist(
            supplier_df[
                "predicted_supplier_score"
            ],
            bins=20,
            color="steelblue",
            edgecolor="black"
        )

        plt.title(
            "Supplier Score Distribution"
        )

        plt.xlabel(
            "Supplier Score"
        )

        plt.ylabel(
            "Count"
        )

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                static_dir,
                "supplier_hist.png"
            )
        )

        plt.close()

    # ====================================================
    # Supplier Trend
    # ====================================================

    if (
        not supplier_df.empty
        and "predicted_supplier_score"
        in supplier_df.columns
    ):

        plt.figure(
            figsize=(10, 5)
        )

        supplier_df[
            "predicted_supplier_score"
        ].rolling(
            20,
            min_periods=1
        ).mean().plot()

        plt.title(
            "Supplier Quality Trend"
        )

        plt.xlabel(
            "Record Index"
        )

        plt.ylabel(
            "Average Supplier Score"
        )

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                static_dir,
                "supplier_curve.png"
            )
        )

        plt.close()

    # ====================================================
    # Fraud Distribution
    # ====================================================

    if (
        not fraud_df.empty
        and "fraud_probability"
        in fraud_df.columns
    ):

        plt.figure(
            figsize=(10, 5)
        )

        plt.hist(
            fraud_df[
                "fraud_probability"
            ],
            bins=20,
            color="red",
            edgecolor="black"
        )

        plt.title(
            "Fraud Probability Distribution"
        )

        plt.xlabel(
            "Fraud Probability"
        )

        plt.ylabel(
            "Count"
        )

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                static_dir,
                "fraud_hist.png"
            )
        )

        plt.close()

    # ====================================================
    # Sensor Anomaly Analysis
    # ====================================================

    if (
        not sensor_df.empty
        and "supplier_id" in sensor_df.columns
        and "anomaly_flag" in sensor_df.columns
    ):

        anomaly_counts = (
            sensor_df[
                sensor_df["anomaly_flag"] == -1
            ]
            .groupby(
                "supplier_id"
            )
            .size()
            .sort_values(
                ascending=False
            )
            .head(10)
        )

        if len(anomaly_counts) > 0:

            plt.figure(
                figsize=(12, 5)
            )

            anomaly_counts.plot(
                kind="bar",
                color="orange"
            )

            plt.title(
                "Top Sensor Anomaly Suppliers"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    static_dir,
                    "sensor_hist.png"
                )
            )

            plt.close()

    # ====================================================
    # RCA Histogram
    # ====================================================

    if (
        not rca_df.empty
        and "rca_score"
        in rca_df.columns
    ):

        plt.figure(
            figsize=(10, 5)
        )

        plt.hist(
            rca_df[
                "rca_score"
            ],
            bins=20,
            color="purple",
            edgecolor="black"
        )

        plt.title(
            "RCA Score Distribution"
        )

        plt.xlabel(
            "RCA Score"
        )

        plt.ylabel(
            "Count"
        )

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                static_dir,
                "rca_hist.png"
            )
        )

        plt.close()

    # ====================================================
    # Top Risk Suppliers
    # ====================================================

    if (
        not supplier_df.empty
        and "predicted_supplier_score"
        in supplier_df.columns
        and "supplier_id"
        in supplier_df.columns
    ):

        top_risk = (
            supplier_df
            .sort_values(
                "predicted_supplier_score"
            )
            .head(10)
        )

        plt.figure(
            figsize=(12, 5)
        )

        plt.bar(
            top_risk[
                "supplier_id"
            ].astype(str),
            top_risk[
                "predicted_supplier_score"
            ],
            color="crimson"
        )

        plt.title(
            "Top 10 Risk Suppliers"
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                static_dir,
                "top_risk_suppliers.png"
            )
        )

        plt.close()
    
# ============================================================
# DATABASE
# ============================================================
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def ensure_column(db, table_name, column_name, column_type):
    info = db.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {row[1] for row in info}
    if column_name not in existing:
        db.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        db.commit()


def init_db():
    db = get_db()

    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            display_name TEXT NOT NULL,
            active_session_token TEXT,
            last_login TEXT,
            created_at TEXT NOT NULL
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            created_by_id INTEGER,
            created_by_name TEXT,

            test_date TEXT,
            plant_name TEXT,
            vehicle_model TEXT,
            supplier_id TEXT,
            supplier_name TEXT,

            part_name TEXT,
            part_category TEXT,
            importance_band TEXT,
            data_source_type TEXT,

            market_price_inr REAL,
            qty_inspected REAL,
            qty_defective REAL,
            ppm REAL,
            otd_pct REAL,
            audit_score_pct REAL,
            cpk REAL,
            criticality_weight_0_1 REAL,
            min_required_sqm_0_1 REAL,
            impact_raw REAL,

            additional_metrics_json TEXT,
            part_specific_metric_0_1 REAL,
            predicted_supplier_quality_score_0_1 REAL,
            predicted_risk_label TEXT,
            predicted_sqm_status TEXT,

            dealer_recommendation TEXT,
            recommendation_reason TEXT,

            preventive_action_statement TEXT,
            dealer_decision TEXT,
            dealer_decision_notes TEXT,

            company_suggestion TEXT,
            company_remarks TEXT,
            company_product_rating_1_10 INTEGER,
            company_updated_at TEXT,

            scoring_source TEXT
        )
    """)
    db.commit()

    ensure_column(db, "records", "part_specific_metric_0_1", "REAL")
    ensure_column(db, "records", "company_suggestion", "TEXT")
    ensure_column(db, "records", "company_remarks", "TEXT")
    ensure_column(db, "records", "company_product_rating_1_10", "INTEGER")
    ensure_column(db, "records", "company_updated_at", "TEXT")
    ensure_column(db, "records", "scoring_source", "TEXT")

    demo_users = [
    ("tester", "Tester@123", ROLE_QUALITY, "Quality Tester"),
    ("dealer", "Dealer@123", ROLE_DEALER, "Dealer User"),
    ("company", "Company@123", ROLE_COMPANY, "Company Manufacturing User"),
    ("auditor", "Auditor@123", ROLE_AUDITOR, "Auditor User"),
]
    for username, password, role, display_name in demo_users:
        exists = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if not exists:
            db.execute(
                """
                INSERT INTO users
                (username, password_hash, role, display_name, active_session_token, last_login, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (username, generate_password_hash(password), role, display_name, None, None, now_str())
            )
    db.commit()
    
# ============================================================
# OPTIONAL MODEL LOADING
# ============================================================
def load_model_from_disk():
    global MODEL, FEATURE_COLUMNS, MODEL_INFO
    MODEL = None
    FEATURE_COLUMNS = []
    MODEL_INFO = {}

    if joblib and os.path.exists(MODEL_PATH):
        try:
            MODEL = joblib.load(MODEL_PATH)
        except Exception:
            MODEL = None

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                MODEL_INFO = json.load(f)
            FEATURE_COLUMNS = MODEL_INFO.get("feature_columns", [])
        except Exception:
            MODEL_INFO = {}
            FEATURE_COLUMNS = []

# ============================================================
# AUTH
# ============================================================
def get_current_user():
    user_id = session.get("user_id")
    token = session.get("session_token")

    if not user_id or not token:
        return None

    user = get_db().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        return None

    if user["active_session_token"] != token:
        session.clear()
        return None

    return user


def login_required(role=None, allowed_roles=None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = get_current_user()
            if not user:
                flash("Please sign in to continue.", "warning")
                return redirect(url_for("login"))
            if role and user["role"] != role:
                flash("You are not authorized to access that page.", "danger")
                return redirect(url_for("dashboard"))
            if allowed_roles and user["role"] not in allowed_roles:
                flash("You are not authorized to access that page.", "danger")
                return redirect(url_for("dashboard"))
            g.current_user = user
            return fn(*args, **kwargs)
        return wrapper
    return decorator


@app.before_request
def ensure_session_validity():
    get_current_user()


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp 

# ============================================================
# SCORING LOGIC
# ============================================================
def parse_additional_metrics(raw_text):
    if not raw_text:
        return {}
    raw_text = raw_text.strip()
    if not raw_text:
        return {}
    try:
        obj = json.loads(raw_text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        parsed = {}
        for line in raw_text.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                val = v.strip()
                try:
                    parsed[k.strip()] = float(val)
                except Exception:
                    parsed[k.strip()] = val
        return parsed


def compute_part_metric(part_name, extra_metrics):
    part_name = (part_name or "").strip().lower()

    def g(key, default=0.0):
        return pct_or_zero(extra_metrics.get(key, default))

    if part_name == "v-belt":
        s_t = clamp(1 - g("belt_tension_dev_ratio") / 0.20)
        s_slip = clamp(1 - g("slip_ratio") / 0.08)
        cmax = max(g("crack_density_max", 10), 1e-6)
        s_crack = clamp(1 - g("crack_density") / cmax)
        return round(0.40 * s_t + 0.35 * s_slip + 0.25 * s_crack, 4)

    if part_name == "radiator hose":
        s_pressure = clamp(g("pressure_retention_ratio"))
        lmax = max(g("leak_rate_max", 2.0), 1e-6)
        s_leak = clamp(1 - g("leak_rate") / lmax)
        s_temp = clamp(1 - g("temp_stability_dev"))
        return round(0.45 * s_pressure + 0.30 * s_leak + 0.25 * s_temp, 4)

    if part_name == "air filter":
        dpmax = max(g("restriction_dp_max", 30), 1e-6)
        s_restriction = clamp(1 - g("restriction_dp") / dpmax)
        eta = g("filtration_efficiency")
        eta_min = g("efficiency_min", 0.95)
        s_eff = clamp((eta - eta_min) / max(1 - eta_min, 1e-6))
        fpmax = max(g("fuel_penalty_max", 0.08), 1e-6)
        s_fuel = clamp(1 - g("fuel_penalty_ratio") / fpmax)
        return round(0.50 * s_restriction + 0.35 * s_eff + 0.15 * s_fuel, 4)

    if part_name == "brake chamber":
        lbmax = max(g("brake_leak_rate_max", 1.0), 1e-6)
        s_leak = clamp(1 - g("brake_leak_rate") / lbmax)
        stroke = g("pushrod_stroke")
        slimit = max(g("stroke_limit", 30), 1e-6)
        s_stroke = clamp(1 - max(0, stroke - slimit) / slimit)
        taumax = max(g("response_lag_max", 0.5), 1e-6)
        s_resp = clamp(1 - g("response_lag") / taumax)
        return round(0.40 * s_leak + 0.35 * s_stroke + 0.25 * s_resp, 4)

    if part_name == "relay valve":
        taumax = max(g("relay_lag_max", 0.4), 1e-6)
        s_lag = clamp(1 - g("relay_lag") / taumax)
        rtarget = max(g("pressure_target_ratio", 1.0), 1e-6)
        s_pressure = clamp(g("pressure_delivery_ratio") / rtarget)
        lmax = max(g("relay_leak_rate_max", 1.0), 1e-6)
        s_leak = clamp(1 - g("relay_leak_rate") / lmax)
        return round(0.45 * s_lag + 0.35 * s_pressure + 0.20 * s_leak, 4)

    if part_name == "wheel speed sensor":
        s_pulse = clamp(g("pulse_integrity"))
        dmax = max(g("speed_dev_max", 0.10), 1e-6)
        s_dev = clamp(1 - g("speed_dev_ratio") / dmax)
        kmax = max(g("dropout_max", 5), 1e-6)
        s_drop = clamp(1 - g("dropout_count") / kmax)
        return round(0.45 * s_pulse + 0.35 * s_dev + 0.20 * s_drop, 4)

    if part_name == "wheel bearing":
        tmax = max(g("bearing_temp_max", 45), 1e-6)
        s_temp = clamp(1 - g("bearing_temp_rise") / tmax)
        vmax = max(g("bearing_vibration_max", 4.0), 1e-6)
        s_vib = clamp(1 - g("bearing_vibration_rms") / vmax)
        pmax = max(g("bearing_play_max", 0.30), 1e-6)
        s_play = clamp(1 - g("bearing_play") / pmax)
        return round(0.40 * s_temp + 0.35 * s_vib + 0.25 * s_play, 4)

    if part_name == "u-joint":
        vmax = max(g("ujoint_vibration_max", 5.0), 1e-6)
        s_vib = clamp(1 - g("ujoint_vibration_rms") / vmax)
        bmax = max(g("ujoint_backlash_max", 0.50), 1e-6)
        s_backlash = clamp(1 - g("ujoint_backlash") / bmax)
        tmax = max(g("ujoint_temp_max", 40), 1e-6)
        s_temp = clamp(1 - g("ujoint_temp_rise") / tmax)
        return round(0.45 * s_vib + 0.30 * s_backlash + 0.25 * s_temp, 4)

    if part_name == "fuel/water separator":
        wmax = max(g("water_out_max", 100), 1e-6)
        s_water = clamp(1 - g("water_out_ppm") / wmax)
        dpmax = max(g("separator_dp_max", 25), 1e-6)
        s_dp = clamp(1 - g("separator_dp") / dpmax)
        s_service = clamp(g("service_compliance_ratio"))
        return round(0.45 * s_water + 0.35 * s_dp + 0.20 * s_service, 4)

    qty_inspected = max(pct_or_zero(extra_metrics.get("qty_inspected", 0)), 1.0)
    qty_defective = max(pct_or_zero(extra_metrics.get("qty_defective", 0)), 0.0)
    defect_rate = clamp(qty_defective / qty_inspected)
    return round(1 - defect_rate, 4)


def derive_risk(score: float, min_required: float, watchlist_band: float = WATCHLIST_BAND) -> str:
    gap = score - min_required
    if gap >= 0:
        return "Low Risk"
    if gap >= -watchlist_band:
        return "Watchlist"
    return "High Risk"


def dealer_suggested_outcome(risk_label, sqm_status, importance_band, criticality, cpk):
    importance_band = (importance_band or "").strip().lower()
    criticality = pct_or_zero(criticality)
    cpk = pct_or_zero(cpk)

    if risk_label == "High Risk" or sqm_status != "Target Met":
        return (
            "Do Not Use",
            "Supplier score is below target or risk is high. Hold/quarantine and re-inspect before dealer usage."
        )
    if risk_label == "Watchlist":
        if importance_band in ("critical", "high") or criticality >= 0.80:
            return (
                "Conditional Use",
                "Important component with watchlist risk. Use only with containment, extra inspection, and approval."
            )
        return (
            "Conditional Use",
            "Use is possible, but monitoring and incoming inspection must continue for the next lots."
        )
    if importance_band == "critical" and cpk < 1.33:
        return (
            "Conditional Use",
            "Low risk but critical component and borderline capability. Prefer controlled use until process capability improves."
        )
    return (
        "Use",
        "Supplier score is acceptable, risk is low, and this supplier can be preferred for this component in current conditions."
    )


def predict_outcome(form_data):
    base_features = {k: form_data.get(k) for k in CORE_COLUMNS}
    for k in NUMERIC_COLUMNS:
        base_features[k] = pct_or_zero(base_features.get(k))

    parsed_metrics = parse_additional_metrics(form_data.get("additional_metrics_json", ""))
    parsed_metrics.setdefault("qty_inspected", base_features.get("qty_inspected", 0))
    parsed_metrics.setdefault("qty_defective", base_features.get("qty_defective", 0))

    part_metric = compute_part_metric(form_data.get("part_name"), parsed_metrics)

    q_ppm = clamp(1 - pct_or_zero(form_data.get("ppm")) / 5000.0)
    q_otd = clamp(pct_or_zero(form_data.get("otd_pct")) / 100.0)
    q_audit = clamp(pct_or_zero(form_data.get("audit_score_pct")) / 100.0)
    q_cpk = clamp(pct_or_zero(form_data.get("cpk")) / 1.33)

    score = round(
        0.25 * q_ppm +
        0.20 * q_otd +
        0.15 * q_audit +
        0.15 * q_cpk +
        0.25 * part_metric,
        4
    )

    scoring_source = "Part-Specific KPI Rule"
    if MODEL is not None and FEATURE_COLUMNS:
        try:
            row_df = pd.DataFrame([{**base_features, **parsed_metrics, "part_specific_metric_0_1": part_metric}])
            for c in FEATURE_COLUMNS:
                if c not in row_df.columns:
                    row_df[c] = np.nan
            model_score = round(float(MODEL.predict(row_df[FEATURE_COLUMNS])[0]), 4)
            score = round((score * 0.60) + (model_score * 0.40), 4)
            scoring_source = "Hybrid: Part Rule + ML Model"
        except Exception:
            scoring_source = "Part-Specific KPI Rule"

    min_required = pct_or_zero(form_data.get("min_required_sqm_0_1"))
    risk_label = derive_risk(score, min_required)
    sqm_status = "Target Met" if score >= min_required else "Below/Need Review"

    recommendation, reason = dealer_suggested_outcome(
        risk_label,
        sqm_status,
        form_data.get("importance_band", ""),
        pct_or_zero(form_data.get("criticality_weight_0_1")),
        pct_or_zero(form_data.get("cpk")),
    )

    return score, risk_label, sqm_status, recommendation, reason, scoring_source, part_metric


def explain_score_breakdown(record_or_form):
    data = dict(record_or_form)
    metrics = parse_additional_metrics(data.get("additional_metrics_json", ""))
    part_metric = compute_part_metric(data.get("part_name"), metrics)
    q_ppm = clamp(1 - pct_or_zero(data.get("ppm")) / 5000.0)
    q_otd = clamp(pct_or_zero(data.get("otd_pct")) / 100.0)
    q_audit = clamp(pct_or_zero(data.get("audit_score_pct")) / 100.0)
    q_cpk = clamp(pct_or_zero(data.get("cpk")) / 1.33)
    score = round(0.25*q_ppm + 0.20*q_otd + 0.15*q_audit + 0.15*q_cpk + 0.25*part_metric, 4)
    return {
        "q_ppm": round(q_ppm, 4),
        "q_otd": round(q_otd, 4),
        "q_audit": round(q_audit, 4),
        "q_cpk": round(q_cpk, 4),
        "part_metric": round(part_metric, 4),
        "rule_score": round(score, 4),
    }

# ============================================================
# DATA HELPERS
# ============================================================
def insert_record(form_data, creator_row):
    score, risk_label, sqm_status, recommendation, reason, scoring_source, part_metric = predict_outcome(form_data)
    db = get_db()

    chosen_part = form_data.get("part_name", "")
    chosen_category = form_data.get("part_category", "") or find_part_category(chosen_part)

    db.execute(
        """
        INSERT INTO records (
            created_at, updated_at, created_by_id, created_by_name,
            test_date, plant_name, vehicle_model, supplier_id, supplier_name,
            part_name, part_category, importance_band, data_source_type,
            market_price_inr, qty_inspected, qty_defective, ppm, otd_pct,
            audit_score_pct, cpk, criticality_weight_0_1, min_required_sqm_0_1,
            impact_raw, additional_metrics_json,
            part_specific_metric_0_1,
            predicted_supplier_quality_score_0_1, predicted_risk_label, predicted_sqm_status,
            dealer_recommendation, recommendation_reason,
            preventive_action_statement, dealer_decision, dealer_decision_notes,
            company_suggestion, company_remarks, company_product_rating_1_10, company_updated_at,
            scoring_source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now_str(), now_str(), creator_row["id"], creator_row["display_name"],
            form_data.get("test_date", ""),
            form_data.get("plant_name", ""),
            form_data.get("vehicle_model", ""),
            form_data.get("supplier_id", ""),
            form_data.get("supplier_name", ""),
            chosen_part,
            chosen_category,
            form_data.get("importance_band", ""),
            form_data.get("data_source_type", ""),
            pct_or_zero(form_data.get("market_price_inr")),
            pct_or_zero(form_data.get("qty_inspected")),
            pct_or_zero(form_data.get("qty_defective")),
            pct_or_zero(form_data.get("ppm")),
            pct_or_zero(form_data.get("otd_pct")),
            pct_or_zero(form_data.get("audit_score_pct")),
            pct_or_zero(form_data.get("cpk")),
            pct_or_zero(form_data.get("criticality_weight_0_1")),
            pct_or_zero(form_data.get("min_required_sqm_0_1")),
            pct_or_zero(form_data.get("impact_raw")),
            form_data.get("additional_metrics_json", ""),
            part_metric,
            score,
            risk_label,
            sqm_status,
            recommendation,
            reason,
            form_data.get("preventive_action_statement", "").strip(),
            "",
            "",
            "",
            "",
            None,
            "",
            scoring_source,
        )
    )
    db.commit()


def fetch_all_records(limit=None):
    sql = "SELECT * FROM records ORDER BY id DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return get_db().execute(sql).fetchall()


def fetch_record(record_id):
    return get_db().execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()


def integrate_historical_dataframe(df):
    if df is None or df.empty:
        return {"records": [], "stats": {"total": 0, "high_risk": 0, "watchlist": 0, "low_risk": 0, "avg_score": 0.0, "avg_part_metric": 0.0}}

    work = df.copy()
    columns_needed = [
        "part_name", "part_category", "ppm", "otd_pct", "audit_score_pct", "cpk",
        "criticality_weight_0_1", "min_required_sqm_0_1", "additional_metrics_json",
        "qty_inspected", "qty_defective", "supplier_name", "supplier_id",
        "vehicle_model", "test_date"
    ]
    for c in columns_needed:
        if c not in work.columns:
            work[c] = ""

    if "predicted_supplier_quality_score_0_1" not in work.columns:
        scores, risks, sqm_statuses, recs, reasons, pm = [], [], [], [], [], []
        for _, row in work.iterrows():
            form_data = row.to_dict()
            score, risk, sqm_status, rec, reason, _, part_metric = predict_outcome(form_data)
            scores.append(score)
            risks.append(risk)
            sqm_statuses.append(sqm_status)
            recs.append(rec)
            reasons.append(reason)
            pm.append(part_metric)
        work["predicted_supplier_quality_score_0_1"] = scores
        work["predicted_risk_label"] = risks
        work["predicted_sqm_status"] = sqm_statuses
        work["dealer_recommendation"] = recs
        work["recommendation_reason"] = reasons
        work["part_specific_metric_0_1"] = pm
    else:
        if "part_specific_metric_0_1" not in work.columns:
            work["part_specific_metric_0_1"] = 0.0
        if "predicted_risk_label" not in work.columns:
            work["predicted_risk_label"] = "Watchlist"
        if "predicted_sqm_status" not in work.columns:
            work["predicted_sqm_status"] = "Below/Need Review"
        if "dealer_recommendation" not in work.columns:
            work["dealer_recommendation"] = "Conditional Use"
        if "recommendation_reason" not in work.columns:
            work["recommendation_reason"] = "Historical dataset view"

    stats = {
        "total": int(len(work)),
        "high_risk": int((work["predicted_risk_label"] == "High Risk").sum()),
        "watchlist": int((work["predicted_risk_label"] == "Watchlist").sum()),
        "low_risk": int((work["predicted_risk_label"] == "Low Risk").sum()),
        "avg_score": round(float(pd.to_numeric(work["predicted_supplier_quality_score_0_1"], errors="coerce").fillna(0).mean()), 4),
        "avg_part_metric": round(float(pd.to_numeric(work["part_specific_metric_0_1"], errors="coerce").fillna(0).mean()), 4),
    }
    records = work.fillna("").to_dict(orient="records")
    return {"records": records, "stats": stats}


def dashboard_data(source="recent"):
    source = (source or "recent").strip().lower()
    if source == "older":
        if os.path.exists(OLDER_DATA_CSV):
            try:
                older_df = pd.read_csv(OLDER_DATA_CSV)
                return integrate_historical_dataframe(older_df)
            except Exception:
                flash("Unable to read testing.csv. Showing recent data instead.", "warning")
        else:
            flash("testing.csv not found. Showing recent data instead.", "warning")

    records = fetch_all_records()
    df = pd.DataFrame([dict(r) for r in records]) if records else pd.DataFrame()

    if df.empty:
        return {
            "records": [],
            "stats": {
                "total": 0,
                "high_risk": 0,
                "watchlist": 0,
                "low_risk": 0,
                "avg_score": 0.0,
                "avg_part_metric": 0.0,
            }
        }

    stats = {
        "total": int(len(df)),
        "high_risk": int((df["predicted_risk_label"] == "High Risk").sum()),
        "watchlist": int((df["predicted_risk_label"] == "Watchlist").sum()),
        "low_risk": int((df["predicted_risk_label"] == "Low Risk").sum()),
        "avg_score": round(float(df["predicted_supplier_quality_score_0_1"].fillna(0).mean()), 4),
        "avg_part_metric": round(float(df.get("part_specific_metric_0_1", pd.Series([0] * len(df))).fillna(0).mean()), 4),
    }

    return {"records": [dict(r) for r in records], "stats": stats}


# ============================================================
# SESSION DRAFT HELPERS
# ============================================================
def get_part_metric_draft():
    return session.get("part_metric_draft", {})


def clear_part_metric_draft():
    session.pop("part_metric_draft", None)


def save_part_metric_draft(part_name, metrics):
    session["part_metric_draft"] = {
        "part_name": part_name,
        "part_category": find_part_category(part_name),
        "metrics": metrics,
        "saved_at": now_str(),
    }


def escape(s):
    return html.escape(str(s or ""))


# ============================================================
# PDF GENERATION
# ============================================================
def build_pdf_bytes(record):
    if SimpleDocTemplate is None:
        raise RuntimeError("reportlab is not installed. Please run: py -m pip install reportlab")

    from io import BytesIO
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm, topMargin=14*mm, bottomMargin=14*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Meta", fontSize=9, leading=12, textColor=colors.HexColor("#4a5a72"), alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="Body2", fontSize=10, leading=14, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="Heading2Blue", fontSize=12, leading=16, textColor=colors.HexColor("#0a4d8c"), spaceAfter=6, spaceBefore=8))

    rec = dict(record)
    metrics = parse_additional_metrics(rec.get("additional_metrics_json") or "")
    breakdown = explain_score_breakdown(rec)

    story = []
    story.append(Paragraph(f"<b>{REPORT_TITLE}</b>", styles["Title"]))
    story.append(Paragraph(f"Record ID: {rec.get('id')} | Generated at: {now_str()}", styles["Meta"]))
    story.append(Paragraph(f"Supplier: <b>{escape(rec.get('supplier_name'))}</b> ({escape(rec.get('supplier_id'))}) | Part: <b>{escape(rec.get('part_name'))}</b>", styles["Meta"]))
    story.append(Spacer(1, 7))

    summary = [
        ["Test Date", escape(rec.get("test_date")), "Plant", escape(rec.get("plant_name"))],
        ["Vehicle Model", escape(rec.get("vehicle_model")), "Category", escape(rec.get("part_category"))],
        ["Scoring Source", escape(rec.get("scoring_source") or "Part-Specific KPI Rule"), "Importance Band", escape(rec.get("importance_band"))],
        ["Part Metric", f"{pct_or_zero(rec.get('part_specific_metric_0_1')):.4f}", "Overall SQA Score", f"{pct_or_zero(rec.get('predicted_supplier_quality_score_0_1')):.4f}"],
        ["Risk", escape(rec.get("predicted_risk_label")), "SQM Status", escape(rec.get("predicted_sqm_status"))],
        ["Recommendation", escape(rec.get("dealer_recommendation")), "Min Required SQM", f"{pct_or_zero(rec.get('min_required_sqm_0_1')):.4f}"],
    ]
    t = Table(summary, colWidths=[34*mm, 60*mm, 34*mm, 48*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dfefff")),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#c8d4e6")),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor("#d7e0ec")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f7fbff")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    story.append(Paragraph("KPI Calculation Breakdown", styles["Heading2Blue"]))
    breakdown_rows = [
        ["Qppm = clamp(1 - PPM/5000)", f"{breakdown['q_ppm']:.4f}"],
        ["Qotd = OTD/100", f"{breakdown['q_otd']:.4f}"],
        ["Qaudit = Audit/100", f"{breakdown['q_audit']:.4f}"],
        ["QcPk = clamp(CPK/1.33)", f"{breakdown['q_cpk']:.4f}"],
        ["Qpart (part-specific metric)", f"{breakdown['part_metric']:.4f}"],
        ["Rule Score = 0.25*Qppm + 0.20*Qotd + 0.15*Qaudit + 0.15*QcPk + 0.25*Qpart", f"{breakdown['rule_score']:.4f}"],
    ]
    bt = Table(breakdown_rows, colWidths=[120*mm, 40*mm])
    bt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f1f6fc")),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#c8d4e6")),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor("#d7e0ec")),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#fafcff")]),
    ]))
    story.append(bt)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Raw KPI Inputs", styles["Heading2Blue"]))
    kpi_rows = [
        ["PPM", pct_or_zero(rec.get("ppm")), "OTD %", pct_or_zero(rec.get("otd_pct"))],
        ["Audit Score %", pct_or_zero(rec.get("audit_score_pct")), "CPK", pct_or_zero(rec.get("cpk"))],
        ["Qty Inspected", pct_or_zero(rec.get("qty_inspected")), "Qty Defective", pct_or_zero(rec.get("qty_defective"))],
        ["Criticality Weight", pct_or_zero(rec.get("criticality_weight_0_1")), "Impact Raw", pct_or_zero(rec.get("impact_raw"))],
        ["Market Price INR", pct_or_zero(rec.get("market_price_inr")), "Data Source", escape(rec.get("data_source_type"))],
    ]
    kt = Table(kpi_rows, colWidths=[40*mm, 35*mm, 40*mm, 55*mm])
    kt.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#c8d4e6")),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor("#d7e0ec")),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f7fbff")]),
    ]))
    story.append(kt)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Part-Specific Inputs", styles["Heading2Blue"]))
    if metrics:
        metric_rows = [[k, escape(v)] for k, v in metrics.items()]
        mt = Table(metric_rows, colWidths=[82*mm, 88*mm])
        mt.setStyle(TableStyle([
            ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#c8d4e6")),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor("#d7e0ec")),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f7fbff")]),
        ]))
        story.append(mt)
    else:
        story.append(Paragraph("No part-specific metrics were captured for this record.", styles["Body2"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Quality Manager / Tester Statement", styles["Heading2Blue"]))
    story.append(Paragraph(escape(rec.get("preventive_action_statement") or "No preventive statement added."), styles["Body2"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Verification Notes", styles["Heading2Blue"]))
    notes = (
        f"Company Suggestion: {escape(rec.get('company_suggestion') or 'Pending')}<br/>"
        f"Company Remarks: {escape(rec.get('company_remarks') or 'No company remarks yet.')}<br/>"
        f"Company Product Rating: {escape(rec.get('company_product_rating_1_10') or '—')} / 10<br/>"
        f"Dealer Decision: {escape(rec.get('dealer_decision') or 'Pending')}<br/>"
        f"Dealer Decision Notes: {escape(rec.get('dealer_decision_notes') or 'No dealer notes yet.')}"
    )
    story.append(Paragraph(notes, styles["Body2"]))

    doc.build(story)
    data = buf.getvalue()
    buf.close()
    return data

# ============================================================
# UI TEMPLATE
# ============================================================
BASE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{{ title }}</title>
  <style>
    :root{
      --bg:#07111f; --bg2:#0c1728; --card:#111c31; --line:rgba(255,255,255,.08);
      --text:#ebf4ff; --muted:#9bb0c9; --primary:#5ea8ff; --primary2:#79f0ff; --success:#39d98a;
      --warning:#ffb548; --danger:#ff6b6b; --shadow:0 20px 60px rgba(0,0,0,.30);
    }
    *{box-sizing:border-box}
    html,body{margin:0;padding:0;font-family:Inter,Segoe UI,Arial,sans-serif;background:radial-gradient(circle at top left,#10203a 0%,var(--bg) 45%,#050a14 100%);color:var(--text);min-height:100vh}
    a{text-decoration:none;color:inherit}
    .app-shell{display:flex;min-height:100vh}
    .overlay{position:fixed;inset:0;background:rgba(0,0,0,.48);opacity:0;pointer-events:none;transition:.25s;z-index:20}
    body.sidebar-open .overlay{opacity:1;pointer-events:auto}
    .sidebar{position:fixed;left:0;top:0;bottom:0;width:290px;padding:22px;background:linear-gradient(180deg,rgba(21,34,61,.98),rgba(13,22,40,.96));border-right:1px solid var(--line);box-shadow:var(--shadow);transform:translateX(-105%);transition:transform .28s ease;z-index:30;display:flex;flex-direction:column;justify-content:space-between}
    body.sidebar-open .sidebar{transform:translateX(0)}
    .content{flex:1;width:100%;padding:20px 22px 30px}
    .topbar{display:flex;justify-content:space-between;align-items:center;gap:16px;padding:12px 14px;border:1px solid var(--line);border-radius:20px;background:rgba(17,28,49,.82);backdrop-filter:blur(14px);box-shadow:var(--shadow);position:sticky;top:12px;z-index:10}
    .top-left,.user-right,.user-card,.brand,.button-group,.pill-row,.form-actions,.split-header{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
    .hamburger{width:42px;height:42px;border:none;border-radius:14px;cursor:pointer;background:linear-gradient(135deg,var(--primary),var(--primary2));color:#072238;font-size:20px;font-weight:900}
    .title-wrap h1,.brand h2,.auth-header h1{margin:0}
    .title-wrap small,.small,.muted,.footer-note,.hint{color:var(--muted)}
    .small,.footer-note,.hint{font-size:12px}
    .user-card{padding:10px 14px;border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.04)}
    .avatar,.brand-mark,.auth-logo{display:grid;place-items:center}
    .avatar{width:38px;height:38px;border-radius:12px;background:rgba(94,168,255,.18);color:var(--primary2);font-weight:800}
    .brand-mark,.auth-logo{
     display:grid;
     place-items:center;
     overflow:hidden;
     background:linear-gradient(135deg,var(--primary),var(--primary2));
     color:#072238;
     font-weight:900;
    }
    .brand-mark{
     width:52px;
     height:52px;
     border-radius:16px;
    }
    .auth-logo{
     width:82px;
     height:82px;
     border-radius:24px;
     margin:0 auto 14px;
     font-size:24px;
    }

    .logo-img{
     width:100%;
     height:100%;
     object-fit:cover;
     display:block;
    }
    .role-badge,.pill,.risk-pill,.decision-pill,.rate-pill{display:inline-flex;padding:6px 11px;border-radius:999px;font-size:12px;font-weight:800}
    .role-quality_tester{background:rgba(94,168,255,.18);color:#bfe1ff}
    .role-dealer{background:rgba(255,181,72,.18);color:#ffd89b}
    .role-company_user{background:rgba(57,217,138,.18);color:#bff5d8}
    .nav-links{display:grid;gap:10px}
    .nav-link{padding:12px 14px;border-radius:14px;background:rgba(255,255,255,.03);border:1px solid transparent;color:var(--muted)}
    .nav-link:hover{color:var(--text);border-color:var(--line);background:rgba(255,255,255,.06)}
    .section-title{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:18px 0 10px}
    .mini,.box,.info-block,.kpi,.table-card,.form-card,.card,.auth-card,.live-card{border:1px solid var(--line);background:linear-gradient(180deg,rgba(21,34,61,.96),rgba(13,22,40,.92));box-shadow:var(--shadow)}
    .mini,.box,.info-block,.live-card{padding:14px;border-radius:16px}
    .card,.table-card,.form-card,.auth-card{border-radius:24px;padding:22px}
    .live-card{position:sticky;top:95px}
    .mini b{display:block;font-size:22px;margin-top:6px}
    .btn{display:inline-flex;justify-content:center;align-items:center;gap:8px;padding:12px 16px;border-radius:14px;border:1px solid transparent;font-weight:800;cursor:pointer;color:var(--text);background:rgba(255,255,255,.05)}
    .btn-primary{background:linear-gradient(135deg,var(--primary),var(--primary2));color:#08233a}
    .btn-ghost{background:rgba(255,255,255,.05);border-color:var(--line)}
    .btn-danger{background:linear-gradient(135deg,#ff6b6b,#ff8f8f);color:#fff}
    .btn-small{padding:8px 12px;font-size:12px}
    .btn-outline{background:transparent;border-color:var(--line)}
    .btn-success{background:linear-gradient(135deg,#1bb56d,#3ee19a);color:#08233a}
    .btn-no{background:linear-gradient(135deg,#ff7b7b,#ff4e4e);color:white}
    .flash-stack{display:grid;gap:10px;margin:18px 0}.flash{padding:14px 16px;border-radius:14px;border:1px solid var(--line)}
    .flash-success{background:rgba(57,217,138,.14)} .flash-danger{background:rgba(255,107,107,.14)} .flash-warning{background:rgba(255,181,72,.14)} .flash-info{background:rgba(94,168,255,.14)}
    .auth-wrap{min-height:calc(100vh - 40px);display:grid;place-items:center}
    .auth-card{width:min(700px,92vw);padding:34px}
    .auth-header{text-align:center;margin-bottom:22px}
    .auth-grid,.form-grid,.detail-grid,.info-grid,.kpi-grid,.hero,.split-grid{display:grid;gap:16px}
    .auth-grid,.form-grid,.info-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
    .hero{grid-template-columns:1.3fr .7fr;margin:20px 0}
    .detail-grid{grid-template-columns:1.2fr 1fr}
    .split-grid{grid-template-columns:2fr 1fr}
    .kpi-grid{grid-template-columns:repeat(6,minmax(0,1fr));margin:18px 0}
    .hero-main,.hero-side{padding:22px}.hero-main h2{margin:0 0 8px;font-size:34px;line-height:1.05}
    .field{display:flex;flex-direction:column;gap:8px;margin-bottom:14px}
    .field label{font-size:13px;font-weight:800;color:#d9e8ff}
    input,textarea,select{width:100%;background:rgba(255,255,255,.06);color:var(--text);border:1px solid rgba(255,255,255,.10);padding:12px 14px;border-radius:14px;outline:none}
    input:focus,textarea:focus,select:focus{border-color:var(--primary);box-shadow:0 0 0 3px rgba(94,168,255,.15)}
    textarea{min-height:110px;resize:vertical}.full{grid-column:1 / -1}
    .table-wrap{overflow:auto} table{width:100%;border-collapse:collapse} th,td{padding:14px 12px;border-bottom:1px solid var(--line);vertical-align:top} th{text-align:left;color:#d5e5ff;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
    tr:hover td{background:rgba(255,255,255,.03)} .low-risk,.use,.yes{background:rgba(57,217,138,.18);color:#9ff0c7} .watchlist,.conditional-use{background:rgba(255,181,72,.18);color:#ffd89b} .high-risk,.do-not-use,.no{background:rgba(255,107,107,.18);color:#ffc0c0} .need-retest{background:rgba(94,168,255,.18);color:#bfe1ff}
    .mt-20{margin-top:20px}
    .metric-list{display:grid;gap:8px;margin-top:12px}
    .metric-item{padding:10px 12px;border-radius:12px;background:rgba(255,255,255,.04);border:1px solid var(--line)}
    .value-large{font-size:32px;font-weight:900;line-height:1}
    .score-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:12px}
    .score-cell{padding:12px;border-radius:14px;background:rgba(255,255,255,.04);border:1px solid var(--line)}
    .subtle-card{padding:14px;border-radius:16px;background:rgba(255,255,255,.04);border:1px solid var(--line)}
    @media(max-width:1100px){.hero,.detail-grid,.form-grid,.auth-grid,.kpi-grid,.info-grid,.split-grid{grid-template-columns:1fr!important}.content{padding:16px}.topbar{top:8px}.live-card{position:static}}
  </style>
  <script>
    function toggleSidebar(){ document.body.classList.toggle('sidebar-open'); }
    function closeSidebar(){ document.body.classList.remove('sidebar-open'); }
    const partOptions = {{ part_options_json|safe }};
    const partSpecConfig = {{ part_spec_json|safe }};
    const partMetricDraft = {{ part_metric_draft_json|safe }};

    function jsClamp(value, low=0, high=1){
      const n = Number(value);
      if(isNaN(n)){ return low; }
      return Math.max(low, Math.min(n, high));
    }
    function jsNum(value, fallback=0){
      const n = Number(value);
      return isNaN(n) ? fallback : n;
    }
    function updatePartOptions(){
      const cat = document.getElementById('part_category');
      const part = document.getElementById('part_name');
      if(!cat || !part){ return; }
      const selected = cat.value;
      const currentValue = part.dataset.current || part.value || '';
      part.innerHTML = '<option value="">Select part</option>';
      (partOptions[selected] || []).forEach(function(item){
        const opt = document.createElement('option');
        opt.value = item;
        opt.textContent = item;
        if(item === currentValue){ opt.selected = true; }
        part.appendChild(opt);
      });
      syncMetricButtonState();
      updateMetricDraftPanel();
      recomputeLiveScore();
    }
    function syncPartCategoryFromName(){
      const part = document.getElementById('part_name');
      const cat = document.getElementById('part_category');
      if(!part || !cat){ return; }
      const currentPart = part.value;
      Object.keys(partOptions).forEach(function(k){
        if((partOptions[k] || []).includes(currentPart)){ cat.value = k; }
      });
      syncMetricButtonState();
      updateMetricDraftPanel();
      recomputeLiveScore();
    }
    function syncMetricButtonState(){
      const btn = document.getElementById('btn-part-metric-page');
      const part = document.getElementById('part_name');
      if(btn && part){ btn.disabled = !part.value; btn.style.opacity = part.value ? '1' : '.55'; }
    }
    function openPartMetricPage(){
      const part = document.getElementById('part_name');
      const cat = document.getElementById('part_category');
      if(!part || !part.value){
        alert('Please select Part Category and Part Name first.');
        return;
      }
      window.location.href = '/part-metrics-config?part_name=' + encodeURIComponent(part.value) + '&part_category=' + encodeURIComponent(cat ? cat.value : '');
    }
    function getMetricDataForCurrentPart(){
      const part = document.getElementById('part_name');
      const partValue = part ? part.value : '';
      if(partMetricDraft && partMetricDraft.part_name === partValue){ return partMetricDraft.metrics || {}; }
      return {};
    }
    function computePartMetricJs(partName, extraMetrics){
      partName = (partName || '').trim().toLowerCase();
      const g = (key, def=0) => jsNum(extraMetrics[key], def);
      if(partName === 'v-belt'){
        const sT = jsClamp(1 - g('belt_tension_dev_ratio') / 0.20);
        const sSlip = jsClamp(1 - g('slip_ratio') / 0.08);
        const cmax = Math.max(g('crack_density_max', 10), 1e-6);
        const sCrack = jsClamp(1 - g('crack_density') / cmax);
        return +(0.40*sT + 0.35*sSlip + 0.25*sCrack).toFixed(4);
      }
      if(partName === 'radiator hose'){
        const sPressure = jsClamp(g('pressure_retention_ratio'));
        const lmax = Math.max(g('leak_rate_max', 2.0), 1e-6);
        const sLeak = jsClamp(1 - g('leak_rate') / lmax);
        const sTemp = jsClamp(1 - g('temp_stability_dev'));
        return +(0.45*sPressure + 0.30*sLeak + 0.25*sTemp).toFixed(4);
      }
      if(partName === 'air filter'){
        const dpmax = Math.max(g('restriction_dp_max', 30), 1e-6);
        const sRestriction = jsClamp(1 - g('restriction_dp') / dpmax);
        const eta = g('filtration_efficiency');
        const etaMin = g('efficiency_min', 0.95);
        const sEff = jsClamp((eta - etaMin) / Math.max(1 - etaMin, 1e-6));
        const fpmax = Math.max(g('fuel_penalty_max', 0.08), 1e-6);
        const sFuel = jsClamp(1 - g('fuel_penalty_ratio') / fpmax);
        return +(0.50*sRestriction + 0.35*sEff + 0.15*sFuel).toFixed(4);
      }
      if(partName === 'brake chamber'){
        const lbmax = Math.max(g('brake_leak_rate_max', 1.0), 1e-6);
        const sLeak = jsClamp(1 - g('brake_leak_rate') / lbmax);
        const stroke = g('pushrod_stroke');
        const slimit = Math.max(g('stroke_limit', 30), 1e-6);
        const sStroke = jsClamp(1 - Math.max(0, stroke - slimit) / slimit);
        const taumax = Math.max(g('response_lag_max', 0.5), 1e-6);
        const sResp = jsClamp(1 - g('response_lag') / taumax);
        return +(0.40*sLeak + 0.35*sStroke + 0.25*sResp).toFixed(4);
      }
      if(partName === 'relay valve'){
        const taumax = Math.max(g('relay_lag_max', 0.4), 1e-6);
        const sLag = jsClamp(1 - g('relay_lag') / taumax);
        const rtarget = Math.max(g('pressure_target_ratio', 1.0), 1e-6);
        const sPressure = jsClamp(g('pressure_delivery_ratio') / rtarget);
        const lmax = Math.max(g('relay_leak_rate_max', 1.0), 1e-6);
        const sLeak = jsClamp(1 - g('relay_leak_rate') / lmax);
        return +(0.45*sLag + 0.35*sPressure + 0.20*sLeak).toFixed(4);
      }
      if(partName === 'wheel speed sensor'){
        const sPulse = jsClamp(g('pulse_integrity'));
        const dmax = Math.max(g('speed_dev_max', 0.10), 1e-6);
        const sDev = jsClamp(1 - g('speed_dev_ratio') / dmax);
        const kmax = Math.max(g('dropout_max', 5), 1e-6);
        const sDrop = jsClamp(1 - g('dropout_count') / kmax);
        return +(0.45*sPulse + 0.35*sDev + 0.20*sDrop).toFixed(4);
      }
      if(partName === 'wheel bearing'){
        const tmax = Math.max(g('bearing_temp_max', 45), 1e-6);
        const sTemp = jsClamp(1 - g('bearing_temp_rise') / tmax);
        const vmax = Math.max(g('bearing_vibration_max', 4.0), 1e-6);
        const sVib = jsClamp(1 - g('bearing_vibration_rms') / vmax);
        const pmax = Math.max(g('bearing_play_max', 0.30), 1e-6);
        const sPlay = jsClamp(1 - g('bearing_play') / pmax);
        return +(0.40*sTemp + 0.35*sVib + 0.25*sPlay).toFixed(4);
      }
      if(partName === 'u-joint'){
        const vmax = Math.max(g('ujoint_vibration_max', 5.0), 1e-6);
        const sVib = jsClamp(1 - g('ujoint_vibration_rms') / vmax);
        const bmax = Math.max(g('ujoint_backlash_max', 0.50), 1e-6);
        const sBacklash = jsClamp(1 - g('ujoint_backlash') / bmax);
        const tmax = Math.max(g('ujoint_temp_max', 40), 1e-6);
        const sTemp = jsClamp(1 - g('ujoint_temp_rise') / tmax);
        return +(0.45*sVib + 0.30*sBacklash + 0.25*sTemp).toFixed(4);
      }
      if(partName === 'fuel/water separator'){
        const wmax = Math.max(g('water_out_max', 100), 1e-6);
        const sWater = jsClamp(1 - g('water_out_ppm') / wmax);
        const dpmax = Math.max(g('separator_dp_max', 25), 1e-6);
        const sDp = jsClamp(1 - g('separator_dp') / dpmax);
        const sService = jsClamp(g('service_compliance_ratio'));
        return +(0.45*sWater + 0.35*sDp + 0.20*sService).toFixed(4);
      }
      const qtyInspected = Math.max(jsNum(document.querySelector('[name="qty_inspected"]')?.value, 0), 1);
      const qtyDefective = Math.max(jsNum(document.querySelector('[name="qty_defective"]')?.value, 0), 0);
      return +(1 - jsClamp(qtyDefective/qtyInspected)).toFixed(4);
    }
    function getText(elId){ return document.getElementById(elId); }
    function classifyRisk(score, minRequired){
      const gap = score - minRequired;
      if(gap >= 0) return 'Low Risk';
      if(gap >= -0.05) return 'Watchlist';
      return 'High Risk';
    }
    function classifyRecommendation(risk, sqmStatus, importanceBand, criticality, cpk){
      importanceBand = (importanceBand||'').toLowerCase();
      if(risk === 'High Risk' || sqmStatus !== 'Target Met'){ return ['Do Not Use', 'Supplier score is below target or risk is high.']; }
      if(risk === 'Watchlist'){
        if(['critical','high'].includes(importanceBand) || criticality >= 0.80){ return ['Conditional Use', 'Important component with watchlist risk.']; }
        return ['Conditional Use', 'Use possible, but monitoring should continue.'];
      }
      if(importanceBand === 'critical' && cpk < 1.33){ return ['Conditional Use', 'Critical component with borderline capability.']; }
      return ['Use', 'Supplier score is acceptable.'];
    }
    function updateMetricDraftPanel(){
      const part = document.getElementById('part_name');
      const badge = document.getElementById('draft-badge');
      const list = document.getElementById('metric-draft-list');
      const raw = document.getElementById('additional_metrics_json');
      if(!list || !raw){ return; }
      list.innerHTML = '';
      const active = getMetricDataForCurrentPart();
      raw.value = JSON.stringify(active || {});
      if(part && partMetricDraft && partMetricDraft.part_name === part.value && Object.keys(active || {}).length){
        if(badge){ badge.innerText = 'Metrics Saved'; }
        Object.keys(active).forEach(function(k){
          const div = document.createElement('div');
          div.className = 'metric-item';
          div.innerHTML = '<b>' + k + '</b><div class="hint">' + active[k] + '</div>';
          list.appendChild(div);
        });
      }else{
        if(badge){ badge.innerText = 'Pending'; }
        list.innerHTML = '<div class="hint">No specific metric page saved yet for this selected part.</div>';
      }
    }
    function recomputeLiveScore(){
      const partEl = document.getElementById('part_name');
      if(!partEl){ return; }
      const partName = partEl.value || '';
      const partMetrics = getMetricDataForCurrentPart();
      const qppm = jsClamp(1 - jsNum(document.querySelector('[name="ppm"]')?.value, 0) / 5000.0);
      const qotd = jsClamp(jsNum(document.querySelector('[name="otd_pct"]')?.value, 0) / 100.0);
      const qaudit = jsClamp(jsNum(document.querySelector('[name="audit_score_pct"]')?.value, 0) / 100.0);
      const qcpk = jsClamp(jsNum(document.querySelector('[name="cpk"]')?.value, 0) / 1.33);
      const partMetric = computePartMetricJs(partName, partMetrics);
      const score = +(0.25*qppm + 0.20*qotd + 0.15*qaudit + 0.15*qcpk + 0.25*partMetric).toFixed(4);
      const minRequired = jsNum(document.querySelector('[name="min_required_sqm_0_1"]')?.value, 0);
      const risk = classifyRisk(score, minRequired);
      const sqm = score >= minRequired ? 'Target Met' : 'Below/Need Review';
      const rec = classifyRecommendation(risk, sqm, document.querySelector('[name="importance_band"]')?.value || '', jsNum(document.querySelector('[name="criticality_weight_0_1"]')?.value, 0), jsNum(document.querySelector('[name="cpk"]')?.value, 0));
      if(getText('live-q-ppm')) getText('live-q-ppm').innerText = qppm.toFixed(4);
      if(getText('live-q-otd')) getText('live-q-otd').innerText = qotd.toFixed(4);
      if(getText('live-q-audit')) getText('live-q-audit').innerText = qaudit.toFixed(4);
      if(getText('live-q-cpk')) getText('live-q-cpk').innerText = qcpk.toFixed(4);
      if(getText('live-q-part')) getText('live-q-part').innerText = partMetric.toFixed(4);
      if(getText('live-score')) getText('live-score').innerText = score.toFixed(4);
      if(getText('live-risk')) getText('live-risk').innerText = risk;
      if(getText('live-status')) getText('live-status').innerText = sqm;
      if(getText('live-rec')) getText('live-rec').innerText = rec[0];
      if(getText('live-reason')) getText('live-reason').innerText = rec[1];
      if(getText('live-part-name')) getText('live-part-name').innerText = partName || 'Select part';
    }
    function setCompanySuggestion(value){
      const hidden = document.getElementById('company_suggestion');
      if(hidden){ hidden.value = value; }
      const yesBtn = document.getElementById('btn-company-yes');
      const noBtn = document.getElementById('btn-company-no');
      if(yesBtn && noBtn){
        yesBtn.classList.toggle('btn-success', value === 'Yes');
        noBtn.classList.toggle('btn-no', value === 'No');
        yesBtn.classList.toggle('btn-outline', value !== 'Yes');
        noBtn.classList.toggle('btn-outline', value !== 'No');
      }
    }
    document.addEventListener('DOMContentLoaded', function(){
      updatePartOptions();
      syncPartCategoryFromName();
      syncMetricButtonState();
      updateMetricDraftPanel();
      recomputeLiveScore();
      const hidden = document.getElementById('company_suggestion');
      if(hidden && hidden.value){ setCompanySuggestion(hidden.value); }
      document.querySelectorAll('input, select, textarea').forEach(function(el){
        el.addEventListener('input', recomputeLiveScore);
        el.addEventListener('change', function(){
          if(el.id === 'part_name'){ syncPartCategoryFromName(); }
          if(el.id === 'part_category'){ updatePartOptions(); }
          updateMetricDraftPanel();
          recomputeLiveScore();
        });
      });
    });
  </script>
</head>
<body>
  <div class="app-shell">
    {% if user %}
      <div class="overlay" onclick="closeSidebar()"></div>
      <aside class="sidebar">
        <div>
          <div class="brand">
            <div class="brand-mark">
            <img src="/logo" alt="SQA Logo" class="logo-img"></div>
            <div>
              <h2>{{ app_title }}</h2>
              <p>Quality + Company + Dealer Workspace</p>
            </div>
          </div>
          <div class="section-title">Navigation</div>
          <nav class="nav-links">
            <a class="nav-link" href="/dashboard?source=recent">Dashboard</a>
            {% if user['role'] == 'quality_tester' %}
              <a class="nav-link" href="/records/new">New Inspection Record</a>
            {% endif %}
            {% if user['role'] == 'auditor' %}
              <a class="nav-link" href="/auditor-dashboard">Auditor Dashboard</a>{% endif %}
          </nav>
        </div>
        <div class="mini">
          <div class="small">Signed in as</div>
          <b>{{ user['display_name'] }}</b>
          <div class="role-badge role-{{ user['role'] }}">{{ role_label(user['role']) }}</div>
        </div>
      </aside>
    {% endif %}

    <main class="content">
      {% if user %}
        <div class="topbar">
          <div class="top-left">
            <button class="hamburger" onclick="toggleSidebar()">☰</button>
            <div class="title-wrap">
              <h1>{{ app_title }}</h1>
              <small>Clean multi-role supplier quality analysis portal</small>
            </div>
          </div>
          <div class="user-right">
            <div class="user-card">
              <div class="avatar">{{ user['display_name'][:1] }}</div>
              <div>
                <div class="small">{{ role_label(user['role']) }}</div>
                <div class="name">{{ user['display_name'] }}</div>
              </div>
            </div>
            <a class="btn btn-ghost" href="/logout">Logout</a>
          </div>
        </div>
      {% endif %}

      {% with msgs = get_flashed_messages(with_categories=true) %}
        {% if msgs %}
          <div class="flash-stack">
            {% for category, message in msgs %}
              <div class="flash flash-{{ category }}">{{ message }}</div>
            {% endfor %}
          </div>
        {% endif %}
      {% endwith %}

      {{ body|safe }}
    </main>
  </div>
</body>
</html>
"""


def render_page(body: str, title: str = APP_TITLE):
    return render_template_string(
        BASE_HTML,
        app_title=APP_TITLE,
        title=title,
        body=body,
        user=get_current_user(),
        role_label=role_label,
        part_options_json=json.dumps(PART_GROUPS),
        part_spec_json=json.dumps(PART_SPEC_CONFIG),
        part_metric_draft_json=json.dumps(get_part_metric_draft()),
    )

# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    return redirect(url_for("dashboard") if get_current_user() else url_for("login"))


@app.route("/logo")
def logo():
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image1.jpeg")
    if not os.path.exists(logo_path):
        return "Logo image1.jpeg not found", 404
    return send_file(logo_path, mimetype="image/jpeg")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        display_name = request.form.get("display_name", "").strip() or username
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        role = request.form.get("role", ROLE_DEALER).strip()

        if not username or not password:
            flash("Username and password are required.", "danger")
            return redirect(url_for("signup"))
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("signup"))
        if role not in ALLOWED_ROLES:
            flash("Invalid role selected.", "danger")
            return redirect(url_for("signup"))

        db = get_db()
        exists = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if exists:
            flash("Username already exists. Choose another username.", "warning")
            return redirect(url_for("signup"))

        db.execute(
            "INSERT INTO users (username, password_hash, role, display_name, active_session_token, last_login, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (username, generate_password_hash(password), role, display_name, None, None, now_str())
        )
        db.commit()
        flash("Signup successful. Please login.", "success")
        return redirect(url_for("login"))

    body = f"""
    <div class="auth-wrap">
      <div class="auth-card">
        <div class="auth-header">
          <div class="auth-logo">SQA</div>
          <h1>Create account</h1>
          <p class="muted">Choose your portal role and create an account.</p>
        </div>
        <form method="post">
          <div class="auth-grid">
            <div class="field"><label>Display Name</label><input name="display_name" placeholder="Your full name" required></div>
            <div class="field"><label>Username</label><input name="username" placeholder="Unique username" required></div>
            <div class="field"><label>Password</label><input type="password" name="password" required></div>
            <div class="field"><label>Confirm Password</label><input type="password" name="confirm_password" required></div>
            <div class="field full">
              <label>Role</label>
              <select name="role">
                <option value="{ROLE_QUALITY}">Quality Tester</option>
                <option value="{ROLE_COMPANY}">Company Manufacturing User</option>
                <option value="{ROLE_DEALER}">Dealer</option>
                <option value="{ROLE_AUDITOR}">Auditor</option>
              </select>
            </div>
          </div>
          <div class="form-actions">
            <button class="btn btn-primary" type="submit">Sign Up</button>
            <a class="btn btn-ghost" href="/login">Back to Login</a>
          </div>
        </form>
      </div>
    </div>
    """
    return render_page(body, "Sign Up")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid username or password.", "danger")
            return redirect(url_for("login"))

        token = secrets.token_hex(24)
        db.execute(
            "UPDATE users SET active_session_token = ?, last_login = ? WHERE id = ?",
            (token, now_str(), user["id"])
        )
        db.commit()

        session.clear()
        session["user_id"] = user["id"]
        session["session_token"] = token

        flash("Login successful.", "success")

        if user["role"] == ROLE_AUDITOR:
          return redirect(url_for("auditor_dashboard"))

        return redirect(url_for("dashboard"))

    body = f"""
    <div class="auth-wrap">
      <div class="auth-card">
        <div class="auth-header">
          <div class="auth-logo">SQA</div>
          <h1>Supplier Quality Portal</h1>
          <p class="muted">Premium workspace for <strong>Quality Testers</strong>, <strong>Company Manufacturing Users</strong>, and <strong>Dealers</strong>.</p>
        </div>
        <form method="post">
          <div class="field"><label>Username</label><input name="username" placeholder="Enter username" required></div>
          <div class="field"><label>Password</label><input type="password" name="password" placeholder="Enter password" required></div>
          <div class="form-actions">
            <button class="btn btn-primary" type="submit">Secure Login</button>
            <a class="btn btn-ghost" href="/signup">Sign Up</a>
          </div>
        </form>
        <div class="mini mt-20">
  <div class="small">Secure Access</div>
  <div class="footer-note">
      Please login using your registered account.
  </div>
  <div class="footer-note">
      New login from another browser invalidates the old session automatically.
  </div>
</div>
      </div>
    </div>
    """
    return render_page(body, "Login")


@app.route("/logout")
def logout():
    user = get_current_user()
    if user:
        get_db().execute("UPDATE users SET active_session_token = NULL WHERE id = ?", (user["id"],))
        get_db().commit()
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required()
def dashboard():
    source = request.args.get("source", "recent")
    data = dashboard_data(source=source)
    stats = data["stats"]
    records = data["records"]
    current_role = g.current_user["role"]

    company_columns = ""
    if current_role in (ROLE_COMPANY, ROLE_DEALER):
        company_columns = "<th>Company Suggestion</th><th>Company Rating</th><th>PDF</th>"

    rows_html = ""
    for idx, r in enumerate(records):
        if source == "older":
          action_link = f"/older-record/{idx}"
        else:
          action_link = f"/records/{r.get('id',0)}"
        risk_cls = (r.get("predicted_risk_label") or "Watchlist").lower().replace(" ", "-")
        rec_cls = (r.get("dealer_recommendation") or "Conditional Use").lower().replace(" ", "-")
        company_cells = ""
        if current_role in (ROLE_COMPANY, ROLE_DEALER):
            sugg = r.get("company_suggestion") or "Pending"
            sugg_cls = ("yes" if sugg == "Yes" else "no") if sugg in ("Yes", "No") else "need-retest"
            rating = r.get("company_product_rating_1_10") or "—"
            pdf_btn = f"<a class='btn btn-small btn-outline' href='/records/{r.get('id',0)}/pdf'>PDF</a>" if r.get('id') else "—"
            company_cells = f"<td><span class='decision-pill {sugg_cls}'>{sugg}</span><div class='small'>{r.get('company_remarks','') or 'No company remarks yet.'}</div></td><td>{rating}</td><td>{pdf_btn}</td>"

        company_feedback_cell = "" if current_role == ROLE_QUALITY else company_cells
        rows_html += f"""
        <tr>
          <td>#{r.get('id','H')}</td>
          <td><b>{escape(r.get('supplier_name',''))}</b><div class='small'>{escape(r.get('supplier_id',''))}</div></td>
          <td><b>{escape(r.get('part_name',''))}</b><div class='small'>{escape(r.get('part_category',''))}</div></td>
          <td>{pct_or_zero(r.get('part_specific_metric_0_1')):.4f}</td>
          <td>{pct_or_zero(r.get('predicted_supplier_quality_score_0_1')):.4f}</td>
          <td><span class='risk-pill {risk_cls}'>{escape(r.get('predicted_risk_label',''))}</span></td>
          <td>{escape(r.get('predicted_sqm_status',''))}</td>
          <td><span class='decision-pill {rec_cls}'>{escape(r.get('dealer_recommendation',''))}</span><div class='small'>{escape(r.get('recommendation_reason',''))}</div></td>
          {company_feedback_cell}
          <td>{escape(r.get('dealer_decision','Pending') or 'Pending')}</td>
          <td><a class='btn btn-small btn-ghost'href='{action_link}'>Open</a></td>
        </tr>
        """

    if not rows_html:
        colspan = 10 + (3 if current_role in (ROLE_COMPANY, ROLE_DEALER) else 0)
        rows_html = f"<tr><td colspan='{colspan}' class='muted'>No records available in this view yet.</td></tr>"

    body = f"""
    <div class="hero">
      <div class="card hero-main">
        <h2>Shared SQA Outcome Dashboard</h2>
        <p class="muted">Quality Tester captures inspection data, part-specific metrics now open in a dedicated page, and live SQA score updates on-screen before save. Dealer and company users can download a verification PDF containing the calculations and quality statement.</p>
        <div class="pill-row">
          <span class="pill">Part-wise metric input page</span>
          <span class="pill">Live SQA score card</span>
          <span class="pill">Dealer/company PDF verification</span>
          <span class="pill">Company suggestion hidden from tester</span>
        </div>
        <div class="button-group mt-20">
          <a class="btn {'btn-primary' if source == 'recent' else 'btn-ghost'}" href="/dashboard?source=recent">Open Recent Dashboard</a>
          <a class="btn {'btn-primary' if source == 'older' else 'btn-ghost'}" href="/dashboard?source=older">Open Older Dashboard (testing.csv)</a>
        </div>
      </div>
      <div class="card hero-side">
        <div class="mini"><div class="small">Current View</div><b>{'Older Data (testing.csv)' if source == 'older' else 'Recent Portal Data'}</b></div>
        <div class="mini mt-20"><div class="small">Scoring</div><b>Part-Specific + KPI Rule</b></div>
      </div>
    </div>

    <div class="kpi-grid">
      <div class="kpi"><div class="label">Total Records</div><div class="value">{stats['total']}</div></div>
      <div class="kpi danger"><div class="label">High Risk</div><div class="value">{stats['high_risk']}</div></div>
      <div class="kpi warning"><div class="label">Watchlist</div><div class="value">{stats['watchlist']}</div></div>
      <div class="kpi success"><div class="label">Low Risk</div><div class="value">{stats['low_risk']}</div></div>
      <div class="kpi"><div class="label">Avg SQA Score</div><div class="value">{stats['avg_score']:.4f}</div></div>
      <div class="kpi"><div class="label">Avg Part Metric</div><div class="value">{stats['avg_part_metric']:.4f}</div></div>
    </div>

    <div class="table-card">
      <h3 style="margin-top:0">Latest SQA Outcomes</h3>
      <p class="muted">Use the buttons above to switch between recent portal data and older testing.csv data.</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th><th>Supplier</th><th>Part</th><th>Part Metric</th><th>Overall Score</th><th>Risk</th><th>SQM Status</th><th>System Recommendation</th>
              {company_columns}
              <th>Dealer Decision</th><th>Action</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </div>
    """
    return render_page(body, "Dashboard")


@app.route("/part-metrics-config", methods=["GET", "POST"])
@login_required(role=ROLE_QUALITY)
def part_metrics_config():
    part_name = (request.values.get("part_name") or "").strip()
    if not part_name:
        flash("Please select a part first from the inspection page.", "warning")
        return redirect(url_for("new_record"))

    config = PART_SPEC_CONFIG.get(part_name)
    if not config:
        flash("No part-specific metric configuration found for this part.", "warning")
        return redirect(url_for("new_record", part_name=part_name))

    existing = get_part_metric_draft() if get_part_metric_draft().get("part_name") == part_name else {}
    existing_metrics = existing.get("metrics", {}) if existing else {}

    if request.method == "POST":
        metrics = {}
        for field in config["fields"]:
            metrics[field["name"]] = pct_or_zero(request.form.get(field["name"], field.get("default", 0)))
        save_part_metric_draft(part_name, metrics)
        flash(f"Specific metrics saved for {part_name}. Return to the inspection page and submit the record.", "success")
        return redirect(url_for("new_record", part_name=part_name, part_category=find_part_category(part_name)))

    fields_html = ""
    for field in config["fields"]:
        val = existing_metrics.get(field["name"], field.get("default", ""))
        fields_html += f"""
        <div class='field'>
          <label>{escape(field['label'])}</label>
          <input type='{field.get('type','number')}' name='{field['name']}' step='{field.get('step','0.01')}' min='{field.get('min','')}' max='{field.get('max','')}' placeholder='{field.get('placeholder','')}' value='{escape(val)}' required>
        </div>
        """

    preview_metric = compute_part_metric(part_name, existing_metrics) if existing_metrics else 0.0
    body = f"""
    <div class='split-grid'>
      <div class='form-card'>
        <div class='split-header' style='justify-content:space-between;margin-bottom:12px'>
          <div>
            <h2 style='margin:0'>{escape(config['title'])}</h2>
            <p class='muted'>{escape(config['help'])}</p>
          </div>
          <div class='pill'>{escape(part_name)}</div>
        </div>
        <form method='post'>
          <input type='hidden' name='part_name' value='{escape(part_name)}'>
          <div class='form-grid'>
            {fields_html}
          </div>
          <div class='form-actions'>
            <button class='btn btn-primary' type='submit'>Save Specific Metrics</button>
            <a class='btn btn-ghost' href='/records/new?part_name={part_name}&part_category={find_part_category(part_name)}'>Back to Inspection</a>
          </div>
          <div class='mt-20'>

    <a class='btn btn-success'
       href='/company-analytics'>
       Open Manufacturing Analytics
    </a>

</div>
        </form>
      </div>
      <div class='live-card'>
        <div class='small'>Current preview metric</div>
        <div class='value-large'>{preview_metric:.4f}</div>
        <div class='hint mt-20'>Once saved, this part-specific data will be used in the main inspection page and in the final SQA rule score.</div>
        <div class='metric-list'>
          <div class='metric-item'><b>Part</b><div class='hint'>{escape(part_name)}</div></div>
          <div class='metric-item'><b>Category</b><div class='hint'>{escape(find_part_category(part_name))}</div></div>
          <div class='metric-item'><b>Status</b><div class='hint'>{'Saved draft available' if existing_metrics else 'Fill fields and save'}</div></div>
        </div>
      </div>
    </div>
    """
    return render_page(body, f"{part_name} Specific Metrics")

@app.route("/auditor-dashboard")
@login_required(role=ROLE_AUDITOR)
def auditor_dashboard():


    generate_dashboard_charts()

    data = load_auditor_data()

    sensor_df = data["sensor"]
    supplier_df = data["supplier"]
    fraud_df = data["fraud"]
    rca_df = data["rca"]
    total_anomalies = 0
    risk_suppliers = 0
    fraud_cases = 0
    critical_rca = 0

    if (
        not sensor_df.empty
        and "anomaly_flag" in sensor_df.columns
    ):
        total_anomalies = (
            sensor_df["anomaly_flag"] == -1
        ).sum()

    if (
        not supplier_df.empty
        and "predicted_supplier_score"
        in supplier_df.columns
    ):
        risk_suppliers = (
            supplier_df[
                "predicted_supplier_score"
            ] < 0.30
        ).sum()

    if (
        not fraud_df.empty
        and "fraud_probability"
        in fraud_df.columns
    ):
        fraud_cases = (
    fraud_df[
        "fraud_probability"
    ] > 0.80
).sum()

    if (
        not rca_df.empty
        and "rca_score"
        in rca_df.columns
    ):
        critical_rca = (
            rca_df[
                "rca_score"
            ] >= 80
        ).sum()

    body = f"""

    <div class='hero'>

        <div class='card hero-main'>

            <h2>
                Auditor Intelligence Dashboard
            </h2>

            <p class='muted'>
                Unified Manufacturing Analytics Platform
            </p>

        </div>

    </div>

    <div class='kpi-grid'>

        <div class='kpi'>
            <div class='label'>
                Sensor Anomalies
            </div>
            <div class='value'>
                {total_anomalies}
            </div>
        </div>

        <div class='kpi'>
            <div class='label'>
                Risk Suppliers
            </div>
            <div class='value'>
                {risk_suppliers}
            </div>
        </div>

        <div class='kpi'>
            <div class='label'>
                Fraud Alerts
            </div>
            <div class='value'>
                {fraud_cases}
            </div>
        </div>

        <div class='kpi'>
            <div class='label'>
                Critical RCA
            </div>
            <div class='value'>
                {critical_rca}
            </div>
        </div>

    </div>

    <div class='card mt-20'>

    <h3>
        Analytics Modules
    </h3>

    <div class='button-group'>

        <a class='btn'
           href='/auditor/sensor'>
            Sensor Intelligence
        </a>

        <a class='btn'
           href='/auditor/supplier-risk'>
            Supplier Risk
        </a>

        <a class='btn'
           href='/auditor/fraud'>
            Fraud Analysis
        </a>

        <a class='btn'
           href='/auditor/rca'>
            RCA Intelligence
        </a>

    </div>

</div>
    <div class='card mt-20'>

    <h3>
        Supplier Score Histogram
    </h3>

    <img
        src="/static/supplier_hist.png"
        style="
            width:100%;
            border-radius:16px;
        "
    >

</div>

    <div class='card mt-20'>

    <h3>
        Supplier Quality Trend
    </h3>

    <img
        src="/static/supplier_curve.png"
        style="
            width:100%;
            border-radius:16px;
        "
    >

</div>

    <div class='card mt-20'>

    <h3>
        RCA Distribution
    </h3>

    <img
        src="/static/rca_hist.png"
        style="
            width:100%;
            border-radius:16px;
        "
    >

</div>

    """

    return render_page(
        body,
        "Auditor Dashboard"
    )

@app.route("/auditor/sensor")
@login_required(role=ROLE_AUDITOR)
def auditor_sensor():

    df = pd.read_csv(
        "sensor_anomaly_results.csv"
    )

    df = df[
        df["anomaly_flag"] == -1
    ].head(100)

    table = df.to_html(
        classes="table",
        index=False
    )

    return render_page(
        table,
        "Sensor Intelligence"
    )

@app.route("/auditor/supplier-risk")
@login_required(role=ROLE_AUDITOR)
def auditor_supplier_risk():

    df = pd.read_csv(
        "supplier_predictions.csv"
    )

    df = df.sort_values(
        "predicted_supplier_score"
    ).head(100)

    table = df.to_html(
        classes="table",
        index=False
    )

    return render_page(
        table,
        "Supplier Risk"
    )

@app.route("/auditor/fraud")
@login_required(role=ROLE_AUDITOR)
def auditor_fraud():

    df = pd.read_csv(
        "claim_fraud_predictions.csv"
    )

    df = df.sort_values(
        "fraud_probability",
        ascending=False
    ).head(100)

    table = df.to_html(
        classes="table",
        index=False
    )

    return render_page(
        table,
        "Fraud Analysis"
    )

@app.route("/auditor/rca")
@login_required(role=ROLE_AUDITOR)
def auditor_rca():

    df = pd.read_csv(
        "unified_rca_results.csv"
    )

    df = df.sort_values(
        "rca_score",
        ascending=False
    ).head(100)

    table = df.to_html(
        classes="table",
        index=False
    )

    return render_page(
        table,
        "RCA Intelligence"
    )

@app.route(
    "/company-analytics",
    methods=["GET", "POST"]
)
@login_required(
    allowed_roles=[
        ROLE_COMPANY,
        ROLE_AUDITOR
    ]
)
def company_analytics():

    generate_dashboard_charts()

    data = load_auditor_data()

    supplier_df = data["supplier"]
    fraud_df = data["fraud"]
    rca_df = data["rca"]
    sensor_df = data["sensor"]

    top_supplier_table = ""
    top_fraud_table = ""
    top_rca_table = ""

    if (
        not supplier_df.empty
        and "predicted_supplier_score"
        in supplier_df.columns
    ):
        top_supplier_table = (
            supplier_df
            .sort_values(
                "predicted_supplier_score"
            )
            .head(10)
            .to_html(
                classes="table",
                index=False
            )
        )

    if (
        not fraud_df.empty
        and "fraud_probability"
        in fraud_df.columns
    ):
        top_fraud_table = (
            fraud_df
            .sort_values(
                "fraud_probability",
                ascending=False
            )
            .head(10)
            .to_html(
                classes="table",
                index=False
            )
        )

    if (
        not rca_df.empty
        and "rca_score"
        in rca_df.columns
    ):
        top_rca_table = (
            rca_df
            .sort_values(
                "rca_score",
                ascending=False
            )
            .head(10)
            .to_html(
                classes="table",
                index=False
            )
        )

    body = f"""

    <div class='card'>

        <h2>
            Manufacturing Analytics Dashboard
        </h2>

        <p class='muted'>
            Quality Intelligence generated from
            Supplier Risk, Fraud, RCA and Sensor Data
        </p>

    </div>

    <div class='kpi-grid'>

        <div class='kpi'>
            <div class='label'>Suppliers</div>
            <div class='value'>{len(supplier_df)}</div>
        </div>

        <div class='kpi'>
            <div class='label'>Fraud Cases</div>
            <div class='value'>{len(fraud_df)}</div>
        </div>

        <div class='kpi'>
            <div class='label'>RCA Cases</div>
            <div class='value'>{len(rca_df)}</div>
        </div>

        <div class='kpi'>
            <div class='label'>Sensor Records</div>
            <div class='value'>{len(sensor_df)}</div>
        </div>

    </div>

    <div class='card mt-20'>

        <h3>
            Supplier Score Histogram
        </h3>

        <img
            src="/static/supplier_hist.png"
            style="
                width:100%;
                border-radius:16px;
            "
        >

    </div>

    <div class='card mt-20'>

        <h3>
            Supplier Quality Trend
        </h3>

        <img
            src="/static/supplier_curve.png"
            style="
                width:100%;
                border-radius:16px;
            "
        >

    </div>

    <div class='card mt-20'>

        <h3>
            Fraud Distribution
        </h3>

        <img
            src="/static/fraud_hist.png"
            style="
                width:100%;
                border-radius:16px;
            "
        >

    </div>

    <div class='card mt-20'>

        <h3>
            Sensor Anomaly Analysis
        </h3>

        <img
            src="/static/sensor_hist.png"
            style="
                width:100%;
                border-radius:16px;
            "
        >

    </div>

    <div class='card mt-20'>

        <h3>
            RCA Distribution
        </h3>

        <img
            src="/static/rca_hist.png"
            style="
                width:100%;
                border-radius:16px;
            "
        >

    </div>

    <div class='card mt-20'>

    <h3>
        Top 10 Risk Suppliers
    </h3>

    <img
        src="/static/top_risk_suppliers.png"
        style="
            width:100%;
            border-radius:16px;
        "
    >

</div>

    """

    return render_page(
        body,
        "Manufacturing Analytics"
    )


@app.route("/records/new", methods=["GET", "POST"])
@login_required(role=ROLE_QUALITY)
def new_record():
    if request.method == "POST":
        form_data = request.form.to_dict()
        chosen_part = form_data.get("part_name", "")
        draft = get_part_metric_draft()
        if draft and draft.get("part_name") == chosen_part:
            form_data["additional_metrics_json"] = json.dumps(draft.get("metrics", {}))
        else:
            form_data["additional_metrics_json"] = json.dumps({})

        if chosen_part in PART_SPEC_CONFIG and parse_additional_metrics(form_data["additional_metrics_json"]) == {}:
            flash("Please open the specific metrics page and save part-specific values before submitting the inspection.", "warning")
            return redirect(url_for("new_record", part_name=chosen_part, part_category=form_data.get("part_category", "")))

        insert_record(form_data, g.current_user)
        clear_part_metric_draft()
        flash("Inspection record saved and published to the dashboard.", "success")
        return redirect(url_for("dashboard"))

    draft = get_part_metric_draft()
    chosen_part = request.args.get("part_name", draft.get("part_name", ""))
    chosen_category = request.args.get("part_category", draft.get("part_category", find_part_category(chosen_part)))
    category_options = "".join([f"<option value='{escape(k)}' {'selected' if k == chosen_category else ''}>{escape(k)}</option>" for k in PART_GROUPS.keys()])

    draft_html = "<div class='hint'>No specific metric page saved yet for this selected part.</div>"
    if draft and draft.get("part_name") == chosen_part and draft.get("metrics"):
        items = "".join([f"<div class='metric-item'><b>{escape(k)}</b><div class='hint'>{escape(v)}</div></div>" for k, v in draft.get("metrics", {}).items()])
        draft_html = items

    body = f"""
    <div class='split-grid'>
      <div class="form-card">
        <div style="margin-bottom:18px">
          <h2 style="margin:0">New Quality Tester Inspection</h2>
          <p class="muted">General KPI inputs remain here. Specific part measurements now open in a dedicated page and the SQA score is shown live while you type.</p>
        </div>
        <form method="post">
          <input type='hidden' id='additional_metrics_json' name='additional_metrics_json' value=''>
          <div class="form-grid">
            <div class="field"><label>Test Date</label><input type="date" name="test_date" required></div>
            <div class="field"><label>Plant Name</label><input name="plant_name" placeholder="e.g. Chennai Plant" required></div>
            <div class="field"><label>Vehicle Model</label><input name="vehicle_model" placeholder="e.g. BharatBenz 2828"></div>
            <div class="field"><label>Data Source Type</label><input name="data_source_type" value="Field / Incoming Inspection"></div>
            <div class="field"><label>Supplier ID</label><input name="supplier_id" required></div>
            <div class="field"><label>Supplier Name</label><input name="supplier_name" required></div>
            <div class="field"><label>Part Category</label><select name="part_category" id="part_category" onchange="updatePartOptions()" required><option value="">Select category</option>{category_options}</select></div>
            <div class="field"><label>Part Name</label><select name="part_name" id="part_name" data-current="{escape(chosen_part)}" onchange="syncPartCategoryFromName()" required><option value="">Select part</option></select></div>
            <div class="field full">
              <label>Specific Part Metrics</label>
              <div class='button-group'>
                <button type='button' id='btn-part-metric-page' class='btn btn-primary' onclick='openPartMetricPage()'>Open Specific Metrics Page</button>
                <span class='pill' id='draft-badge'>{'Metrics Saved' if draft and draft.get('part_name') == chosen_part and draft.get('metrics') else 'Pending'}</span>
              </div>
              <div class='hint'>Select the part first, then open the dedicated page to enter the exact measurement signals for that part.</div>
              <div class='metric-list' id='metric-draft-list'>{draft_html}</div>
            </div>
            <div class="field"><label>Importance Band</label><select name="importance_band"><option>Critical</option><option>High</option><option selected>Medium</option><option>Low</option></select></div>
            <div class="field"><label>Market Price (INR)</label><input type="number" step="0.01" name="market_price_inr"></div>
            <div class="field"><label>Qty Inspected</label><input type="number" step="1" name="qty_inspected" required></div>
            <div class="field"><label>Qty Defective</label><input type="number" step="1" name="qty_defective" required></div>
            <div class="field"><label>PPM</label><input type="number" step="0.01" name="ppm"></div>
            <div class="field"><label>OTD %</label><input type="number" step="0.01" name="otd_pct"></div>
            <div class="field"><label>Audit Score %</label><input type="number" step="0.01" name="audit_score_pct"></div>
            <div class="field"><label>CPK</label><input type="number" step="0.01" name="cpk"></div>
            <div class="field"><label>Criticality Weight (0 to 1)</label><input type="number" step="0.01" min="0" max="1" name="criticality_weight_0_1"></div>
            <div class="field"><label>Minimum Required SQM (0 to 1)</label><input type="number" step="0.01" min="0" max="1" name="min_required_sqm_0_1" required></div>
            <div class="field"><label>Impact Raw</label><input type="number" step="0.01" name="impact_raw"></div>
            <div class="field full"><label>Preventive Action Statement (Quality Manager / Tester)</label><textarea name="preventive_action_statement" placeholder="Add preventive guidance for next lots and containment actions."></textarea></div>
          </div>
          <div class="form-actions">
            <button class="btn btn-primary" type="submit">Save Outcome to Dashboard</button>
            <a class="btn btn-ghost" href="/dashboard">Cancel</a>
          </div>
        </form>
      </div>
      <div class='live-card'>
        <div class='small'>Live SQA score preview</div>
        <div class='value-large' id='live-score'>0.0000</div>
        <div class='pill-row mt-20'>
          <span class='pill' id='live-risk'>Watchlist</span>
          <span class='pill' id='live-status'>Below/Need Review</span>
        </div>
        <div class='score-grid'>
          <div class='score-cell'><div class='small'>Part</div><b id='live-part-name'>Select part</b></div>
          <div class='score-cell'><div class='small'>Qpart</div><b id='live-q-part'>0.0000</b></div>
          <div class='score-cell'><div class='small'>Qppm</div><b id='live-q-ppm'>0.0000</b></div>
          <div class='score-cell'><div class='small'>Qotd</div><b id='live-q-otd'>0.0000</b></div>
          <div class='score-cell'><div class='small'>Qaudit</div><b id='live-q-audit'>0.0000</b></div>
          <div class='score-cell'><div class='small'>QcPk</div><b id='live-q-cpk'>0.0000</b></div>
        </div>
        <div class='subtle-card mt-20'>
          <div class='small'>Live recommendation</div>
          <b id='live-rec'>Conditional Use</b>
          <div class='hint mt-20' id='live-reason'>Enter KPI values and save part-specific metrics to preview the decision logic.</div>
        </div>
      </div>
    </div>
    """
    return render_page(body, "New Inspection Record")

@app.route("/older-record/<int:index>")
@login_required()
def older_record(index):

    if not os.path.exists(OLDER_DATA_CSV):
        flash("testing.csv not found.", "danger")
        return redirect(url_for("dashboard"))

    try:
        df = pd.read_csv(OLDER_DATA_CSV)
    except Exception:
        flash("Unable to read testing.csv.", "danger")
        return redirect(url_for("dashboard"))

    if index < 0 or index >= len(df):
        flash("Record not found.", "danger")
        return redirect(url_for("dashboard"))

    row = df.iloc[index]

    body = f"""
    <div class='table-card'>

        <h2>Historical Record Details</h2>

        <div class='info-grid'>

            <div class='box'>
                <span>Supplier</span>
                <strong>{row.get('supplier_name', '')}</strong>
            </div>

            <div class='box'>
                <span>Supplier ID</span>
                <strong>{row.get('supplier_id', '')}</strong>
            </div>

            <div class='box'>
                <span>Part Name</span>
                <strong>{row.get('part_name', '')}</strong>
            </div>

            <div class='box'>
                <span>Part Category</span>
                <strong>{row.get('part_category', '')}</strong>
            </div>

            <div class='box'>
                <span>PPM</span>
                <strong>{row.get('ppm', '')}</strong>
            </div>

            <div class='box'>
                <span>OTD %</span>
                <strong>{row.get('otd_pct', '')}</strong>
            </div>

            <div class='box'>
                <span>Audit Score</span>
                <strong>{row.get('audit_score_pct', '')}</strong>
            </div>

            <div class='box'>
                <span>CPK</span>
                <strong>{row.get('cpk', '')}</strong>
            </div>

            <div class='box'>
                <span>Importance Band</span>
                <strong>{row.get('importance_band', '')}</strong>
            </div>

            <div class='box'>
                <span>Minimum SQM</span>
                <strong>{row.get('min_required_sqm_0_1', '')}</strong>
            </div>

        </div>

        <div class='mt-20'>
            <a class='btn btn-primary'
               href='/dashboard?source=older'>
               Back to Historical Dashboard
            </a>
        </div>

    </div>
    """

    return render_page(body, "Historical Record")

@app.route("/records/<int:record_id>", methods=["GET", "POST"])
@login_required()
def record_detail(record_id):
    record = fetch_record(record_id)
    if not record:
        flash("Record not found.", "danger")
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        db = get_db()
        if g.current_user["role"] == ROLE_DEALER:
            decision = request.form.get("dealer_decision", "").strip()
            notes = request.form.get("dealer_decision_notes", "").strip()
            db.execute(
                "UPDATE records SET dealer_decision = ?, dealer_decision_notes = ?, updated_at = ? WHERE id = ?",
                (decision, notes, now_str(), record_id)
            )
            db.commit()
            flash("Dealer decision updated.", "success")
        elif g.current_user["role"] == ROLE_COMPANY:
            suggestion = request.form.get("company_suggestion", "").strip()
            remarks = request.form.get("company_remarks", "").strip()
            rating = request.form.get("company_product_rating_1_10", "").strip()
            rating_val = int(rating) if str(rating).isdigit() else None
            db.execute(
                "UPDATE records SET company_suggestion = ?, company_remarks = ?, company_product_rating_1_10 = ?, company_updated_at = ?, updated_at = ? WHERE id = ?",
                (suggestion, remarks, rating_val, now_str(), now_str(), record_id)
            )
            db.commit()
            flash("Company feedback updated.", "success")
        elif g.current_user["role"] == ROLE_QUALITY:
            statement = request.form.get("preventive_action_statement", "").strip()
            db.execute(
                "UPDATE records SET preventive_action_statement = ?, updated_at = ? WHERE id = ?",
                (statement, now_str(), record_id)
            )
            db.commit()
            flash("Preventive action statement updated.", "success")
        return redirect(url_for("record_detail", record_id=record_id))

    metrics = parse_additional_metrics(record["additional_metrics_json"]) if record["additional_metrics_json"] else {}
    metrics_html = ""
    if metrics:
        items = "".join([f"<div class='metric-item'><b>{escape(k)}</b><div class='hint'>{escape(v)}</div></div>" for k, v in metrics.items()])
        metrics_html = f"""
        <div class='info-block mt-20'>
          <div class='label'>Part Specific Metrics Captured</div>
          <div class='metric-list'>{items}</div>
        </div>
        """

    breakdown = explain_score_breakdown(dict(record))
    breakdown_html = f"""
    <div class='info-block mt-20'>
      <div class='label'>Calculation Breakdown</div>
      <div class='metric-list'>
        <div class='metric-item'><b>Qppm</b><div class='hint'>{breakdown['q_ppm']:.4f}</div></div>
        <div class='metric-item'><b>Qotd</b><div class='hint'>{breakdown['q_otd']:.4f}</div></div>
        <div class='metric-item'><b>Qaudit</b><div class='hint'>{breakdown['q_audit']:.4f}</div></div>
        <div class='metric-item'><b>QcPk</b><div class='hint'>{breakdown['q_cpk']:.4f}</div></div>
        <div class='metric-item'><b>Qpart</b><div class='hint'>{breakdown['part_metric']:.4f}</div></div>
        <div class='metric-item'><b>Rule Score</b><div class='hint'>{breakdown['rule_score']:.4f}</div></div>
      </div>
    </div>
    """

    record_dict = dict(record)
    company_feedback_view = ""
    current_role = g.current_user["role"]

    pdf_button = ""
    if current_role in (ROLE_COMPANY, ROLE_DEALER):
        pdf_button = f"<a class='btn btn-small btn-outline' href='/records/{record_id}/pdf'>Download Verification PDF</a>"

    if current_role in (ROLE_COMPANY, ROLE_DEALER):
        sugg = record_dict.get("company_suggestion") or "Pending"
        company_feedback_view = f"""
        <div class='info-block mt-20'>
          <div class='split-header' style='justify-content:space-between'>
            <div class='label'>Company Manufacturing Feedback</div>
            {pdf_button}
          </div>
          <div><strong>Suggestion:</strong> {escape(sugg)}</div>
          <div class='mt-20'><strong>Remarks:</strong> {escape(record_dict.get('company_remarks') or 'No company remarks yet.')}</div>
          <div class='mt-20'><strong>Product Rating:</strong> {escape(record_dict.get('company_product_rating_1_10') or '—')} / 10</div>
        </div>
        """
    elif current_role == ROLE_QUALITY:
        company_feedback_view = f"""
        <div class='info-block mt-20'>
          <div class='label'>Company Product Rating</div>
          <div>{escape(record_dict.get('company_product_rating_1_10') or 'No company rating yet.')} / 10</div>
        </div>
        """

    if current_role == ROLE_DEALER:
        action_form = f"""
        <div class='form-card mt-20'>
          <h3 style='margin-top:0'>Dealer Decision</h3>
          <div class='muted'>Dealer can see company usage suggestion and remarks, and can download the verification PDF for record-level validation.</div>
          <form method='post'>
            <div class='field'>
              <label>Decision</label>
              <select name='dealer_decision'>
                <option value='Use' {'selected' if (record['dealer_decision'] or '') == 'Use' else ''}>Use</option>
                <option value='Conditional Use' {'selected' if (record['dealer_decision'] or '') == 'Conditional Use' else ''}>Conditional Use</option>
                <option value='Do Not Use' {'selected' if (record['dealer_decision'] or '') == 'Do Not Use' else ''}>Do Not Use</option>
                <option value='Need Retest' {'selected' if (record['dealer_decision'] or '') == 'Need Retest' else ''}>Need Retest</option>
              </select>
            </div>
            <div class='field'>
              <label>Decision Notes</label>
              <textarea name='dealer_decision_notes' placeholder='Add dealer justification and field usage restrictions.'>{escape(record['dealer_decision_notes'] or '')}</textarea>
            </div>
            <div class='form-actions'><button class='btn btn-primary' type='submit'>Save Dealer Decision</button></div>
            <div class='card mt-20'>

    <h3>
        Manufacturing Analytics
    </h3>

    <p class='muted'>
        View Supplier Risk,
        Fraud Analysis,
        RCA Analysis and
        Sensor Intelligence.
    </p>

    <a class='btn btn-success'
       href='/company-analytics'>

       Open Analytics Dashboard

    </a>

</div>
          </form>
        </div>
        """
    elif current_role == ROLE_COMPANY:
        rating_options = ''.join([f"<option value='{i}' {'selected' if str(record['company_product_rating_1_10'] or '') == str(i) else ''}>{i}</option>" for i in range(1, 11)])
        action_form = f"""
        <div class='form-card mt-20'>
          <h3 style='margin-top:0'>Company Manufacturing Feedback</h3>
          <p class='muted'>This is internal company guidance for dealer use. Company suggestion and remarks are visible to dealer and company, not to quality tester.</p>
          <form method='post'>
            <input type='hidden' id='company_suggestion' name='company_suggestion' value='{escape(record['company_suggestion'] or '')}'>
            <div class='field'>
              <label>Suggest Dealer Whether to Use Product</label>
              <div class='button-group'>
                <button type='button' id='btn-company-yes' class='btn btn-outline' onclick="setCompanySuggestion('Yes')">Yes</button>
                <button type='button' id='btn-company-no' class='btn btn-outline' onclick="setCompanySuggestion('No')">No</button>
              </div>
            </div>
            <div class='field'>
              <label>Company Remarks</label>
              <textarea name='company_remarks' placeholder='Add remarks for dealer visibility.'>{escape(record['company_remarks'] or '')}</textarea>
            </div>
            <div class='field'>
              <label>Product Rating for Quality Tester (1-10)</label>
              <select name='company_product_rating_1_10'>
                <option value=''>Select rating</option>
                {rating_options}
              </select>
            </div>
            <div class='form-actions'><button class='btn btn-primary' type='submit'>Save Company Feedback</button></div>
          </form>
        </div>
        """
    else:
        action_form = f"""
        <div class='form-card mt-20'>
          <h3 style='margin-top:0'>Quality Tester Preventive Statement</h3>
          <form method='post'>
            <div class='field'>
              <label>Preventive Action Statement</label>
              <textarea name='preventive_action_statement'>{escape(record['preventive_action_statement'] or '')}</textarea>
            </div>
            <div class='form-actions'><button class='btn btn-primary' type='submit'>Save Prevention Guidance</button></div>
          </form>
        </div>
        <form method='post' action='/records/{record_id}/delete' class='mt-20' onsubmit="return confirm('Delete this record?')">
          <button class='btn btn-danger' type='submit'>Delete Record</button>
        </form>
        """

    risk_cls = (record["predicted_risk_label"] or "Watchlist").lower().replace(" ", "-")
    rec_cls = (record["dealer_recommendation"] or "Conditional Use").lower().replace(" ", "-")

    body = f"""
    <div class='detail-grid'>
      <div class='table-card'>
        <h2 style='margin-top:0'>Record #{record['id']} — {escape(record['part_name'])}</h2>
        <div class='pill-row'>
          <span class='risk-pill {risk_cls}'>{escape(record['predicted_risk_label'])}</span>
          <span class='pill'>Part Metric: {pct_or_zero(record['part_specific_metric_0_1']):.4f}</span>
          <span class='pill'>Overall Score: {pct_or_zero(record['predicted_supplier_quality_score_0_1']):.4f}</span>
          <span class='decision-pill {rec_cls}'>{escape(record['dealer_recommendation'])}</span>
          {pdf_button}
        </div>
        <div class='info-grid mt-20'>
          <div class='box'><span>Supplier</span><strong>{escape(record['supplier_name'])} ({escape(record['supplier_id'])})</strong></div>
          <div class='box'><span>Part</span><strong>{escape(record['part_name'])}</strong></div>
          <div class='box'><span>Part Category</span><strong>{escape(record['part_category'])}</strong></div>
          <div class='box'><span>Vehicle Model</span><strong>{escape(record['vehicle_model'])}</strong></div>
          <div class='box'><span>PPM</span><strong>{escape(record['ppm'])}</strong></div>
          <div class='box'><span>OTD %</span><strong>{escape(record['otd_pct'])}</strong></div>
          <div class='box'><span>Audit Score %</span><strong>{escape(record['audit_score_pct'])}</strong></div>
          <div class='box'><span>CPK</span><strong>{escape(record['cpk'])}</strong></div>
          <div class='box'><span>Criticality</span><strong>{escape(record['criticality_weight_0_1'])}</strong></div>
          <div class='box'><span>Min Required SQM</span><strong>{escape(record['min_required_sqm_0_1'])}</strong></div>
        </div>
        <div class='info-block mt-20'><div class='label'>Recommendation Reason</div><div>{escape(record['recommendation_reason'] or '')}</div></div>
        <div class='info-block mt-20'><div class='label'>Preventive Action Statement</div><div>{escape(record['preventive_action_statement'] or 'No preventive statement added yet.')}</div></div>
        {breakdown_html}
        {company_feedback_view}
        {metrics_html}
      </div>
      <div>{action_form}</div>
    </div>
    """
    return render_page(body, f"Record #{record_id}")


@app.route("/records/<int:record_id>/pdf")
@login_required(allowed_roles=[ROLE_DEALER, ROLE_COMPANY])
def record_pdf(record_id):
    record = fetch_record(record_id)
    if not record:
        flash("Record not found.", "danger")
        return redirect(url_for("dashboard"))
    try:
        pdf_data = build_pdf_bytes(record)
    except Exception as ex:
        flash(str(ex), "danger")
        return redirect(url_for("record_detail", record_id=record_id))

    response = make_response(pdf_data)
    response.headers["Content-Type"] = "application/pdf"
    filename = f"sqa_record_{record_id}_verification_report.pdf"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


@app.route("/records/<int:record_id>/delete", methods=["POST"])
@login_required(role=ROLE_QUALITY)
def delete_record(record_id):
    db = get_db()
    db.execute("DELETE FROM records WHERE id = ?", (record_id,))
    db.commit()
    flash("Record deleted.", "info")
    return redirect(url_for("dashboard"))


# ============================================================
# STARTUP
# ============================================================
with app.app_context():
    init_db()

load_model_from_disk()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)



