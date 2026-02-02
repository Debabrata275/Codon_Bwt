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

try:
    codon_df = pd.read_csv("codon_usage.csv", low_memory=False)
    codon_df.columns = [c.strip().upper() for c in codon_df.columns]

    # Convert all non-metadata columns to numeric safely
    for col in codon_df.columns:
        if col not in ["SPECIESNAME", "KINGDOM"]:
            codon_df[col] = pd.to_numeric(codon_df[col], errors="coerce")

except Exception as e:
    raise RuntimeError(f"Failed to load codon_usage.csv: {e}")

# ================= LOAD ML MODEL (SAFE) =================

feature_importance = {}

try:
    model = joblib.load("model_outputs/global_codon_bwt_model.pkl")
    booster = model.get_booster()
    feature_importance = booster.get_score(importance_type="weight")
except Exception as e:
    print("⚠ ML model not loaded:", e)

# ================= LOAD EVALUATION METRICS =================

try:
    with open("model_outputs/evaluation_metrics.json", "r") as f:
        EVAL_METRICS = json.load(f)
except Exception:
    EVAL_METRICS = {}

# ================= GENETIC CODE =================

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

table = CodonTable.unambiguous_rna_by_id[1]
AA_TO_CODONS = {}
for codon, aa in table.forward_table.items():
    AA_TO_CODONS.setdefault(aa, []).append(codon)

# ================= CORE ANALYSIS =================

def analyze_core(aa, selected_codon=None):
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
    df_tmp["SCORE"] = df_tmp[selected_codon] if selected_codon in codons else df_tmp[codons].sum(axis=1)

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
        codons = [c for c in AA_TO_CODONS.get(aa, []) if c in codon_df.columns]
        if not codons:
            return None

        df = codon_df.copy()
        df["PREFERENCE_SCORE"] = df[codon] if codon in codons else df[codons].sum(axis=1)

        max_val = df["PREFERENCE_SCORE"].max()
        if max_val and max_val > 0:
            df["PREFERENCE_SCORE"] /= max_val

        return {
            "top_species": df.sort_values("PREFERENCE_SCORE", ascending=False)
                .head(5)[["SPECIESNAME", "PREFERENCE_SCORE"]]
                .to_dict(orient="records"),
            "bottom_species": df.sort_values("PREFERENCE_SCORE")
                .head(5)[["SPECIESNAME", "PREFERENCE_SCORE"]]
                .to_dict(orient="records"),
            "explanation": f"Species-specific preference variation for amino acid {aa}."
        }
    except Exception as e:
        print("Species analysis failed:", e)
        return None

# ================= HOST-AWARE OPTIMIZATION =================

def host_aware_optimization(aa, host_species):
    try:
        if not host_species:
            return None

        codons = [c for c in AA_TO_CODONS.get(aa, []) if c in codon_df.columns]
        df = codon_df[codon_df["SPECIESNAME"].str.contains(host_species, case=False, na=False)]

        if df.empty or not codons:
            return None

        mean_usage = df[codons].mean(skipna=True).sort_values(ascending=False)

        return {
            "host_species": host_species,
            "optimal_codon": mean_usage.index[0],
            "codon_ranking": list(mean_usage.items())
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
        print("Bias score failed:", e)
        return None

# ================= CROSS-KINGDOM COMPARISON =================

def cross_kingdom_comparison(codon):
    try:
        if "KINGDOM" not in codon_df.columns or codon not in codon_df.columns:
            return []

        grouped = (
            codon_df.groupby("KINGDOM")[codon]
            .mean(skipna=True)
            .reset_index()
            .dropna()
        )
        return grouped.to_dict(orient="records")
    except Exception as e:
        print("Kingdom comparison failed:", e)
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
    host_species = data.get("host_species", "")

    result, error = analyze_core(aa, codon)
    if error:
        return jsonify({"error": error}), 400

    return jsonify({
        **result,
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
