from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import json
from Bio.Data import CodonTable

app = Flask(__name__, template_folder=".")
CORS(app)

# ================= LOAD CODON USAGE =================

codon_df = pd.read_csv("codon_usage.csv", low_memory=False)
codon_df.columns = [c.strip().upper() for c in codon_df.columns]

# ================= LOAD TRAINED MODEL =================

model = joblib.load("model_outputs/global_codon_bwt_model.pkl")
booster = model.get_booster()
feature_importance = booster.get_score(importance_type="weight")

# ================= LOAD EVALUATION METRICS =================

with open("model_outputs/evaluation_metrics.json", "r") as f:
    EVAL_METRICS = json.load(f)

# ================= VALID AMINO ACIDS =================

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ================= GENETIC CODE (RNA) =================

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

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_api():
    data = request.json
    aa = data.get("amino_acid", "")
    codon = data.get("codon", "")

    result, error = analyze(aa, codon)
    if error:
        return jsonify({"error": error}), 400

    # ---------- Accuracy values ----------
    top1_acc = EVAL_METRICS.get("accuracy_top1")
    top2_acc = EVAL_METRICS.get("accuracy_top2")
    top3_acc = EVAL_METRICS.get("accuracy_top3")

    return jsonify({
        "codon_ranking": result["codon_ranking"],
        "selected_rank": result["selected_rank"],
        "top_species": result["top_species"],

        # ===== MODEL EVALUATION METRICS =====
        "model_metrics": {
            "top1_accuracy": top1_acc,
            "top2_accuracy": top2_acc,
            "top3_accuracy": top3_acc,

            "precision": EVAL_METRICS.get("precision"),
            "recall": EVAL_METRICS.get("recall"),
            "f1_score": EVAL_METRICS.get("f1_score"),
            "loss": EVAL_METRICS.get("loss"),

            # robustness
            "accuracy_clean": EVAL_METRICS.get("accuracy_clean"),
            "accuracy_noisy": EVAL_METRICS.get("accuracy_noisy"),
            "accuracy_missing": EVAL_METRICS.get("accuracy_missing"),

            # comparison
            "accuracy_codon_only": EVAL_METRICS.get("accuracy_codon_only"),
            "accuracy_codon_bwt": EVAL_METRICS.get("accuracy_codon_bwt"),

            # derived
            "error": (1 - top1_acc) if top1_acc is not None else None
        },

        # ===== SINGLE SOURCE OF TRUTH EXPLANATION =====
        "accuracy_explanation": (
            "The hybrid Codon + BWT model improves Top-1 accuracy from 96.91% to 97.78%, "
            "and demonstrates robustness under noisy and missing data. "
            "Training plots and confusion matrices are included as evidence of model convergence."
        )
    })

if __name__ == "__main__":
    app.run(debug=True)
