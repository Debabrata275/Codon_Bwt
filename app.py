import os
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from Bio.Data import CodonTable

# ================= APP SETUP =================

app = Flask(__name__, template_folder=".")
CORS(app)

# ================= LOAD CODON USAGE DATA =================

codon_df = pd.read_csv("codon_usage.csv", low_memory=False)
codon_df.columns = [c.strip().upper() for c in codon_df.columns]

# Force numeric conversion (CRITICAL for production)
for col in codon_df.columns:
    if col not in ["SPECIESNAME", "KINGDOM"]:
        codon_df[col] = pd.to_numeric(codon_df[col], errors="coerce")

# ================= LOAD ML MODEL (SAFE) =================

feature_importance = {}

try:
    model = joblib.load("model_outputs/global_codon_bwt_model.pkl")
    booster = model.get_booster()
    feature_importance = booster.get_score(importance_type="weight")
except Exception as e:
    print("⚠ ML model not loaded:", e)

# ================= LOAD EVALUATION METRICS (SAFE DEFAULTS) =================

DEFAULT_METRICS = {
    "top1_accuracy": 0.0,
    "top2_accuracy": 0.0,
    "top3_accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0,
    "loss": 0.0,
    "accuracy_clean": 0.0,
    "accuracy_noisy": 0.0,
    "accuracy_missing": 0.0,
    "accuracy_codon_only": 0.0,
    "accuracy_codon_bwt": 0.0
}

try:
    with open("model_outputs/evaluation_metrics.json", "r") as f:
        raw = json.load(f)
        EVAL_METRICS = {k: float(raw.get(k, 0.0)) for k in DEFAULT_METRICS}
except Exception:
    EVAL_METRICS = DEFAULT_METRICS.copy()

# ================= GENETIC CODE =================

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

table = CodonTable.unambiguous_rna_by_id[1]
AA_TO_CODONS = {}

for codon, aa in table.forward_table.items():
    AA_TO_CODONS.setdefault(aa, []).append(codon)

# ================= CORE ANALYSIS =================

def analyze(aa, selected_codon=None):
    aa = aa.upper()
    selected_codon = selected_codon.upper() if selected_codon else None

    if aa not in VALID_AA:
        return None, "Invalid amino acid (use single-letter code like L, K, F)"

    codons = [c for c in AA_TO_CODONS.get(aa, []) if c in codon_df.columns]
    if not codons:
        return None, "No codon data available for this amino acid"

    mean_usage = codon_df[codons].mean(skipna=True).sort_values(ascending=False)

    codon_ranking = []
    selected_rank = None

    for i, codon in enumerate(mean_usage.index, start=1):
        if selected_codon == codon:
            selected_rank = i

        codon_ranking.append({
            "rank": i,
            "codon": codon,
            "frequency": float(mean_usage[codon]) if pd.notna(mean_usage[codon]) else 0.0,
            "ml_weight": float(feature_importance.get(codon.replace("U", "T"), 0.0))
        })

    df_tmp = codon_df.copy()
    if selected_codon and selected_codon in codons:
        df_tmp["SCORE"] = df_tmp[selected_codon]
    else:
        df_tmp["SCORE"] = df_tmp[codons].sum(axis=1)

    top_species = (
        df_tmp.sort_values("SCORE", ascending=False)
        .head(5)[["SPECIESNAME", "SCORE"]]
        .to_dict(orient="records")
    )

    return {
        "codon_ranking": codon_ranking,
        "selected_rank": selected_rank,
        "top_species": top_species
    }, None

# ================= SPECIES-SPECIFIC PREFERENCE =================

def species_specific_analysis(aa, codon=None):
    try:
        aa = aa.upper()
        codon = codon.upper() if codon else None

        codons = [c for c in AA_TO_CODONS.get(aa, []) if c in codon_df.columns]
        if not codons:
            return None

        df = codon_df.copy()

        if codon and codon in codons:
            df["PREFERENCE_SCORE"] = df[codon]
            used_codons = [codon]
        else:
            df["PREFERENCE_SCORE"] = df[codons].sum(axis=1)
            used_codons = codons

        max_val = df["PREFERENCE_SCORE"].max()
        if max_val and max_val > 0:
            df["PREFERENCE_SCORE"] /= max_val

        return {
            "used_codons": used_codons,
            "top_species": df.sort_values("PREFERENCE_SCORE", ascending=False)
                .head(5)[["SPECIESNAME", "PREFERENCE_SCORE"]]
                .to_dict(orient="records"),
            "bottom_species": df.sort_values("PREFERENCE_SCORE")
                .head(5)[["SPECIESNAME", "PREFERENCE_SCORE"]]
                .to_dict(orient="records"),
            "explanation": (
                f"Species-specific codon preference highlights how organisms "
                f"differ in using codon(s) {', '.join(used_codons)} for amino acid {aa}."
            )
        }
    except Exception as e:
        print("Species preference failed:", e)
        return None

# ================= HOST-AWARE OPTIMIZATION =================

def host_aware_optimization(aa, host_species):
    try:
        if not host_species:
            return None

        aa = aa.upper()
        codons = [c for c in AA_TO_CODONS.get(aa, []) if c in codon_df.columns]
        if not codons:
            return None

        df = codon_df[
            codon_df["SPECIESNAME"].str.contains(host_species, case=False, na=False)
        ]

        if df.empty:
            return None

        mean_usage = df[codons].mean(skipna=True).sort_values(ascending=False)

        return {
            "host_species": host_species,
            "optimal_codon": mean_usage.index[0],
            "codon_ranking": [(c, float(v)) for c, v in mean_usage.items()]
        }
    except Exception as e:
        print("Host optimization failed:", e)
        return None

# ================= CODON BIAS SCORE =================

def codon_bias_score(codon):
    try:
        if not codon or codon not in codon_df.columns:
            return None

        col = codon_df[codon]
        global_avg = col.mean(skipna=True)

        if pd.isna(global_avg) or global_avg == 0:
            return None

        df = codon_df.copy()
        df["bias"] = col / global_avg

        return {
            "codon": codon,
            "global_average": float(global_avg),
            "top_bias_species": (
                df.dropna(subset=["bias"])
                .sort_values("bias", ascending=False)
                .head(5)[["SPECIESNAME", "bias"]]
                .to_dict(orient="records")
            )
        }
    except Exception as e:
        print("Codon bias failed:", e)
        return None

# ================= CROSS-KINGDOM COMPARISON =================

def cross_kingdom_comparison(codon):
    try:
        if not codon or codon not in codon_df.columns:
            return []

        if "KINGDOM" not in codon_df.columns:
            return []

        grouped = (
            codon_df.groupby("KINGDOM")[codon]
            .mean(skipna=True)
            .reset_index()
            .dropna()
        )

        return grouped.to_dict(orient="records")
    except Exception as e:
        print("Cross-kingdom failed:", e)
        return []

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_api():
    data = request.json or {}

    aa = data.get("amino_acid", "")
    codon = data.get("codon", "")
    host_species = data.get("host_species", "").strip()

    result, error = analyze(aa, codon)
    if error:
        return jsonify({"error": error}), 400

    return jsonify({
        "codon_ranking": result["codon_ranking"],
        "selected_rank": result["selected_rank"],
        "top_species": result["top_species"],

        "species_specific_analysis": species_specific_analysis(aa, codon),
        "host_aware_optimization": host_aware_optimization(aa, host_species),
        "codon_bias_score": codon_bias_score(codon),
        "cross_kingdom_comparison": cross_kingdom_comparison(codon),

        "model_metrics": EVAL_METRICS
    })

# ================= RUN (RENDER SAFE) =================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
