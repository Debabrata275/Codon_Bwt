from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import json
from Bio.Data import CodonTable

# ================= APP SETUP =================

app = Flask(__name__, template_folder=".")
CORS(app)

# ================= LOAD CODON USAGE DATA =================

codon_df = pd.read_csv("codon_usage.csv", low_memory=False)
codon_df.columns = [c.strip().upper() for c in codon_df.columns]

# ================= LOAD ML MODEL =================

model = joblib.load("model_outputs/global_codon_bwt_model.pkl")
booster = model.get_booster()
feature_importance = booster.get_score(importance_type="weight")

# ================= LOAD EVALUATION METRICS =================

with open("model_outputs/evaluation_metrics.json", "r") as f:
    EVAL_METRICS = json.load(f)

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

    codons = [c for c in AA_TO_CODONS[aa] if c in codon_df.columns]
    if not codons:
        return None, "No codon data available for this amino acid"

    # ---------- Codon Ranking ----------
    mean_usage = codon_df[codons].mean().sort_values(ascending=False)

    codon_ranking = []
    selected_rank = None

    for i, codon in enumerate(mean_usage.index, start=1):
        if selected_codon and codon == selected_codon:
            selected_rank = i

        codon_ranking.append({
            "rank": i,
            "codon": codon,
            "frequency": float(mean_usage[codon]),
            "ml_weight": float(feature_importance.get(codon.replace("U", "T"), 0.0))
        })

    # ---------- Species Ranking ----------
    df_tmp = codon_df.copy()
    if selected_codon and selected_codon in codons:
        df_tmp["SCORE"] = df_tmp[selected_codon]
    else:
        df_tmp["SCORE"] = df_tmp[codons].sum(axis=1)

    top_species = (
        df_tmp.sort_values(by="SCORE", ascending=False)
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
    if max_val > 0:
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

# ================= HOST-AWARE OPTIMIZATION =================

def host_aware_optimization(aa, host_species):
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

    mean_usage = df[codons].mean().sort_values(ascending=False)

    return {
        "host_species": host_species,
        "optimal_codon": mean_usage.index[0],
        "codon_ranking": list(mean_usage.items())
    }

# ================= CODON BIAS SCORE =================

def codon_bias_score(codon):
    if not codon or codon not in codon_df.columns:
        return None

    global_avg = codon_df[codon].mean()
    if global_avg == 0:
        return None

    df = codon_df.copy()
    df["bias"] = df[codon] / global_avg

    return {
        "codon": codon,
        "global_average": float(global_avg),
        "top_bias_species": (
            df.sort_values("bias", ascending=False)
            .head(5)[["SPECIESNAME", "bias"]]
            .to_dict(orient="records")
        )
    }

# ================= CROSS-KINGDOM COMPARISON =================

def cross_kingdom_comparison(codon):
    if not codon or codon not in codon_df.columns:
        return []

    if "KINGDOM" not in codon_df.columns:
        return []

    grouped = (
        codon_df.groupby("KINGDOM")[codon]
        .mean()
        .reset_index()
    )

    return grouped.to_dict(orient="records")

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_api():
    data = request.json

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

        "model_metrics": {
            "top1_accuracy": EVAL_METRICS.get("accuracy_top1"),
            "top2_accuracy": EVAL_METRICS.get("accuracy_top2"),
            "top3_accuracy": EVAL_METRICS.get("accuracy_top3"),
            "precision": EVAL_METRICS.get("precision"),
            "recall": EVAL_METRICS.get("recall"),
            "f1_score": EVAL_METRICS.get("f1_score"),
            "loss": EVAL_METRICS.get("loss"),
            "accuracy_clean": EVAL_METRICS.get("accuracy_clean"),
            "accuracy_noisy": EVAL_METRICS.get("accuracy_noisy"),
            "accuracy_missing": EVAL_METRICS.get("accuracy_missing"),
            "accuracy_codon_only": EVAL_METRICS.get("accuracy_codon_only"),
            "accuracy_codon_bwt": EVAL_METRICS.get("accuracy_codon_bwt"),
        }
    })

# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True)
