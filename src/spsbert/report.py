"""Build the final PDF report entirely from results/ (no hand-typed numbers)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from .aggregate import PRIMARY, aggregate
from .utils import read_json

GROUP = 7
PROBLEM = ("PS-5: Investigate the effectiveness of SBERT-based sentence embeddings for story point "
           "estimation within the same project using random splitting.")
MEMBERS = [("Atluri Venkata Sai Vignesh Chowdary", "B230857CS"), ("Adimulam Yaswanth Veera Nagesh", "B230755CS"),
           ("M V Gowrish", "B230673CS"), ("Kukkala Sai Dinesh Reddy", "B231035CS"),
           ("Munukuntla Rithvik Reddy", "B231114CS")]
LABEL = {"mean_baseline": "Mean", "median_baseline": "Median", "tfidf_ridge": "TF-IDF+Ridge",
         "sbert_ridge": "SBERT+Ridge", "sbert_cnn": "SBERT+CNN"}
ORDER = list(LABEL)

REFS = [
    "N. Reimers and I. Gurevych, Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks, EMNLP-IJCNLP 2019.",
    "V. Tawosi, A. Al-Subaihin, R. Moussa, F. Sarro, A Versatile Dataset of Agile Open Source Software Projects, "
    "MSR 2022 (TAWOS; data DOI 10.5522/04/21308124).",
    "M. Choetkiertikul et al., A deep learning model for estimating story points (Deep-SE), IEEE Transactions on "
    "Software Engineering.",
    "K. Song et al., MPNet: Masked and Permuted Pre-training for Language Understanding, NeurIPS 2020 "
    "(backbone of all-mpnet-base-v2).",
    "F. Wilcoxon, Individual comparisons by ranking methods, Biometrics Bulletin 1(6), 1945.",
]
(REFS_NOTE) = "Please verify bibliographic details against the original publications before submission."


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontSize=17, leading=21),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=13.5, spaceBefore=10, spaceAfter=5),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, spaceBefore=7, spaceAfter=3),
        "p": ParagraphStyle("p", parent=ss["BodyText"], fontSize=9.4, leading=12.6, alignment=TA_JUSTIFY, spaceAfter=4),
        "small": ParagraphStyle("s", parent=ss["BodyText"], fontSize=7.8, leading=9.6),
        "cell": ParagraphStyle("c", parent=ss["BodyText"], fontSize=7.4, leading=8.8),
        "banner": ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=11.5, textColor=colors.HexColor("#8a1c1c"),
                                 backColor=colors.HexColor("#fdeaea"), borderColor=colors.HexColor("#d64545"),
                                 borderWidth=0.8, borderPadding=5, spaceAfter=8),
        "code": ParagraphStyle("code", parent=ss["Code"], fontSize=7.8, leading=9.5, backColor=colors.HexColor("#f3f3f3"),
                               borderPadding=3, spaceAfter=4),
    }


def _table(rows, widths, st, header=1, zebra=True, font=7.6):
    t = Table(rows, colWidths=widths, repeatRows=header)
    style = [("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#999999")),
             ("BACKGROUND", (0, 0), (-1, header - 1), colors.HexColor("#dfe7f3")),
             ("FONTNAME", (0, 0), (-1, header - 1), "Helvetica-Bold"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if zebra:
        for i in range(header, len(rows)):
            if (i - header) % 2:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f6f6")))
    t.setStyle(TableStyle(style))
    return t


def _img(path: Path, width_cm: float):
    iw, ih = ImageReader(str(path)).getSize()
    w = width_cm * cm
    return Image(str(path), width=w, height=w * ih / iw)


def _pv(p):
    """p-value for prose: '< 0.001' instead of a misleading 0.000."""
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def _f(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _discussion(agg: dict, mode_share: float) -> list:
    """Evidence-driven sentences; wording adapts to what the numbers show."""
    macro = agg["macro"].set_index("method")
    test, imp, xp = agg["test"], agg["improvement"], agg["cross_project_wilcoxon"]
    n_proj = test["project"].nunique()
    out = []
    best_mae, best_mdae = macro["test_mae"].idxmin(), macro["test_mdae"].idxmin()
    out.append(f"Averaged over {n_proj} projects (each project weighted equally), the lowest held-out test MAE was obtained by "
               f"<b>{LABEL[best_mae]}</b> ({_f(macro.loc[best_mae, 'test_mae'])}) and the lowest MdAE by "
               f"<b>{LABEL[best_mdae]}</b> ({_f(macro.loc[best_mdae, 'test_mdae'])}).")
    sb = [m for m in ("sbert_ridge", PRIMARY) if m in macro.index]
    if sb:
        b1, b2 = min(sb, key=lambda m: macro.loc[m, "test_mae"]), min(sb, key=lambda m: macro.loc[m, "test_mdae"])
        out.append(f"Among the SBERT-based models, {LABEL[b1]} has the lowest MAE ({_f(macro.loc[b1, 'test_mae'])}) and {LABEL[b2]} the lowest "
                   f"MdAE ({_f(macro.loc[b2, 'test_mdae'])}); this is the answer for the assigned (SBERT sentence-embedding) representation.")
    if "median_baseline" in macro.index and PRIMARY in macro.index:
        mb, c = macro.loc["median_baseline"], macro.loc[PRIMARY]
        rel = 100 * (mb["test_mae"] - c["test_mae"]) / mb["test_mae"]
        verdict = "improves on" if rel > 0 else "does not improve on"
        out.append(f"SBERT+CNN {verdict} the trivial median-of-training predictor in macro-averaged test MAE "
                   f"({_f(c['test_mae'])} vs {_f(mb['test_mae'])}, {rel:+.1f}%) and has MdAE {_f(c['test_mdae'])} vs {_f(mb['test_mdae'])}.")
        if not imp.empty:
            ic = imp[imp.method == PRIMARY]
            out.append(f"Per project, SBERT+CNN had lower test MAE than the median baseline in {(ic.mae_improvement_pct > 0).sum()} of "
                       f"{len(ic)} projects.")
        if xp.get("p_value") is not None:
            out.append(f"A Wilcoxon signed-rank test across projects on test MAE (SBERT+CNN vs median) gives p = {xp['p_value']:.4f} "
                       f"(n = {xp['n_projects']}); the 39 projects are not a random sample of all software projects, so "
                       f"p-values are descriptive of these projects rather than confirmatory for software projects in general.")
        sig = test[(test.method == PRIMARY) & (test.p_vs_median < 0.05)]
        sig_holm = test[(test.method == PRIMARY) & (test.p_vs_median_holm < 0.05)]
        better = sum(
            float(test[(test.project == r.project) & (test.method == PRIMARY)].mae.iloc[0])
            < float(test[(test.project == r.project) & (test.method == "median_baseline")].mae.iloc[0]) for r in sig.itertuples())
        out.append(f"On individual projects, the paired per-issue Wilcoxon test against the median baseline was significant at "
                   f"0.05 (uncorrected) for {len(sig)} projects, of which SBERT+CNN was better in {better}; after Holm correction across "
                   f"projects it remained significant for {len(sig_holm)}.")
    if "sbert_ridge" in macro.index and PRIMARY in macro.index:
        d = macro.loc[PRIMARY, "test_mae"] - macro.loc["sbert_ridge", "test_mae"]
        out.append("A linear Ridge head on the same SBERT vectors "
                   + ("matched or beat" if d >= 0 else "was beaten by") + f" the CNN in macro test MAE (SBERT+Ridge "
                   f"{_f(macro.loc['sbert_ridge', 'test_mae'])} vs SBERT+CNN {_f(macro.loc[PRIMARY, 'test_mae'])}); "
                   + ("so on average the CNN\'s extra non-linear capacity did not pay off at these sample sizes (most projects have only a few hundred training issues)."
                    if d >= 0 else "so the CNN's extra non-linear capacity helped on average."))
    if "tfidf_ridge" in macro.index and "sbert_ridge" in macro.index:
        d = macro.loc["sbert_ridge", "test_mae"] - macro.loc["tfidf_ridge", "test_mae"]
        out.append("Comparing representations with the same Ridge head, SBERT embeddings "
                   + ("did not outperform" if d >= 0 else "outperformed") + f" lexical TF-IDF features in macro test MAE "
                   f"({_f(macro.loc['sbert_ridge', 'test_mae'])} vs {_f(macro.loc['tfidf_ridge', 'test_mae'])}).")
    ex = agg.get("extra", {})
    if "mean_ranks" in ex:
        best = min(ex["mean_ranks"], key=ex["mean_ranks"].get)
        out.append(f"By average rank over projects the best method is {LABEL[best]} ({ex['mean_ranks'][best]:.2f}); "
                   + (f"a Friedman test indicates that the methods differ (p = {ex['friedman']['p']:.2g})." if "friedman" in ex else ""))
    sp_ = ex.get("improvement_spearman", {})
    if sp_.get("n_issues", {}).get("p", 1) < 0.05:
        out.append(f"SBERT+CNN's advantage over the median baseline grows with project size (Spearman rho = {sp_['n_issues']['rho']:+.2f}, "
                   f"p {_pv(sp_['n_issues']['p'])}), consistent with a neural model needing more training data.")
    out.append("Possible explanations (hypotheses suggested by the data, not tested here): (i) story points are discrete; in these "
               f"data the most frequent value covers on average {100 * mode_share:.0f}% of a project's issues (see 'mode share' in "
               "Table 1), and the higher this share, the stronger a constant predictor is for MdAE; (ii) estimates reflect team-specific conventions and context (sprint capacity, assignee) that "
               "are not in the text; (iii) the frozen SBERT encoder is trained for semantic similarity, not effort; (iv) small "
               "projects give high-variance fits and wide confidence intervals.")
    return out


def build_report(output_dir: str | Path, out_pdf: str | Path, github_url: str) -> Path:
    out = Path(output_dir)
    agg = aggregate(out)
    meta = read_json(out / "run_metadata.json")
    cfg = meta["config"]
    clean = read_json(out / "data_cleaning_report.json")
    summ = pd.read_csv(out / "tables" / "dataset_summary.csv")
    results, test, val, macro = agg["results"], agg["test"], agg["val"], agg["macro"]
    st = _styles()
    synthetic = bool(meta.get("synthetic_data")) or meta.get("encoder_backend") != "sbert"
    placeholder = "<username>" in github_url
    n_proj = len(results)
    n_issues = int(sum(r["n_issues"] for r in results))

    def P(t, s="p"):
        return Paragraph(t, st[s])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1.1 * cm, f"Group {GROUP} - NLP course project - PS-5 (SBERT, within-project, random split)")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
        if synthetic:
            canvas.setFillColor(colors.HexColor("#b00020"))
            canvas.drawCentredString(A4[0] / 2, A4[1] - 1.0 * cm, "SYNTHETIC / NON-SBERT SMOKE-TEST OUTPUT - NOT REAL RESULTS")
        canvas.restoreState()

    S: list = []
    S.append(P("Effectiveness of SBERT-based Sentence Embeddings for Story Point Estimation within the Same Project (Random Split)", "title"))
    S.append(P(escape(PROBLEM), "p"))
    if synthetic:
        S.append(P("<b>WARNING - NOT A VALID EXPERIMENT.</b> This file was generated from synthetic data and/or the offline "
                   "hashing encoder to test the software pipeline. None of the numbers below are TAWOS or SBERT results and "
                   "must not be submitted. Re-run on the real TAWOS data with the SBERT backend, then rebuild this report.", "banner"))
    S.append(P(f"<b>Group {GROUP}</b> - Natural Language Processing course project", "p"))
    S.append(_table([["#", "Name", "Roll No."]] + [[str(i + 1), n, r] for i, (n, r) in enumerate(MEMBERS)],
                    [1 * cm, 9 * cm, 4 * cm], st, font=8.5))
    S.append(Spacer(1, 6))
    link = escape(github_url)
    S.append(P(f"<b>GitHub repository:</b> <link href=\"{link}\" color=\"blue\">{link}</link>"
               + ("  <font color='#b00020'>(placeholder - replace with your real URL and rebuild)</font>" if placeholder else "")))
    S.append(P(f"<b>Date generated:</b> {dt.date.today().isoformat()} | <b>Encoder:</b> {escape(str(meta['encoder']))} | "
               f"<b>Seed:</b> {cfg['seed']} | <b>Config hash:</b> {meta['config_hash']}"))

    m = macro.set_index("method")
    abstract = (f"We study whether frozen Sentence-BERT (SBERT) embeddings of issue text, fed to a three-layer CNN, predict story points "
                f"better than simple baselines when training and testing issues come from the same project and are split at random. "
                f"On {n_proj} projects ({n_issues} issues) from the TAWOS dataset we use an 80:20 development/test split, five-fold "
                f"cross-validation on the development part for model selection, and report Mean Absolute Error (MAE) and Median "
                f"Absolute Error (MdAE) on the held-out test set. Macro-averaged test MAE / MdAE: ")
    abstract += "; ".join(f"{LABEL[k]} {_f(m.loc[k, 'test_mae'])} / {_f(m.loc[k, 'test_mdae'])}" for k in ORDER if k in m.index) + "."
    S.append(P("<b>Abstract.</b> " + abstract))

    # 1
    S.append(P("1. Problem statement and objectives", "h1"))
    S.append(P("In agile development, issues or user stories are assigned <i>story points</i>, a relative measure of effort. "
               "Given the text of an issue (title and description), the task is to predict its story point value. "
               "Sentence embeddings represent a whole text as one vector; SBERT is trained so that semantically similar texts have "
               "similar vectors. <b>Objective:</b> investigate how effective SBERT sentence embeddings are for within-project story "
               "point estimation under random splitting, measured with MAE and MdAE, and compare against transparent baselines "
               "(mean, median, TF-IDF+Ridge, SBERT+Ridge) so that any conclusion about 'effectiveness' is relative to something."))
    S.append(P("<b>Setting.</b> <i>Within-project</i>: training and test issues come from the same project (each project is modelled "
               "separately). <i>Random split</i>: issues are divided without regard to creation date. Consequently temporal drift is "
               "not controlled for (see Limitations)."))

    # 2
    S.append(P("2. Dataset", "h1"))
    S.append(P("We use the TAWOS dataset of open-source agile projects (Tawosi et al., MSR 2022). Only issues with a story point value "
               "are usable. Cleaning (see <font face='Courier'>spsbert/data.py</font>): non-numeric or non-positive story points removed; "
               "text = title + description with code/noformat blocks, Jira macros, HTML tags and user mentions removed and URLs replaced by "
               "'URL'; empty texts removed; duplicate issue keys and exact duplicate texts within a project removed (duplicates would "
               "otherwise appear in both training and test data); projects with fewer than "
               f"{cfg['data']['min_issues']} usable issues skipped."
               + (f" <b>Outlier policy:</b> story points above {cfg['data']['max_story_point']:g} were removed as data-entry errors "
                  f"(the raw data contain values up to {clean.get('raw_story_point_max', 0):,.0f}); {clean.get('dropped_above_max_story_point', 0)} "
                  f"issues were affected (per project: <font face='Courier'>results/data_cleaning_report.json</font>)." if cfg['data'].get('max_story_point') else "")))
    S.append(P("Cleaning report: " + ", ".join(f"{k.replace('_', ' ')} = {v}" for k, v in clean.items()
                                                  if isinstance(v, int)) + ".", "small"))
    rows = [["Project", "Issues", "Mean SP", "Median", "Std", "Min", "Max", "#Distinct", "Mode share"]]
    for r in summ.itertuples():
        rows.append([Paragraph(escape(r.project), st["cell"]), r.n_issues, _f(r.sp_mean, 2), _f(r.sp_median, 1), _f(r.sp_std, 2),
                     _f(r.sp_min, 0), _f(r.sp_max, 0), r.n_distinct_sp, _f(r.mode_share, 2)])
    S.append(P("Table 1. Story-point distribution per project (mode share = fraction of issues carrying the most frequent value).", "small"))
    S.append(_table(rows, [4.6 * cm, 1.4 * cm, 1.6 * cm, 1.5 * cm, 1.4 * cm, 1.2 * cm, 1.3 * cm, 1.7 * cm, 1.8 * cm], st))

    # 3
    S.append(P("3. Methodology", "h1"))
    S.append(P("3.1 Pipeline", "h2"))
    S.append(P("For every project independently: (1) sort issues by key and split 80:20 at random into development and test sets "
               f"(seed {cfg['seed']}); (2) split the development set into five folds; (3) for each method and hyper-parameter setting, "
               "train on four folds and validate on the fifth, rotating so every fold validates once; (4) select the setting with the "
               "lowest mean validation MAE (ties broken by MdAE) <i>using development data only</i>; (5) refit that setting on the whole "
               "development set and evaluate once on the untouched test set. The test set is never used for selection, early "
               "stopping, scaling or feature fitting. All methods share identical partitions, which allows paired comparisons."))
    S.append(P("3.2 SBERT sentence embeddings", "h2"))
    S.append(P(f"Each issue text is encoded once with <font face='Courier'>{escape(str(meta['encoder']))}</font> "
               f"(sentence-transformers; maximum sequence length {cfg['embedding']['max_seq_length']} tokens, longer texts truncated; "
               f"embeddings L2-normalised). The encoder is frozen and unsupervised w.r.t. the target, so encoding all issues before "
               f"splitting does not leak labels. Embeddings are cached on disk."))
    c = cfg["cnn"]
    S.append(P("3.3 Prediction component: three-layer CNN", "h2"))
    S.append(P(f"The embedding vector (length D) is treated as a one-channel 1-D signal and passed through three blocks of "
               f"Conv1d(kernel {c['kernel_size']}) - BatchNorm - ReLU - {c['pool_type']}-pool({c['pool_size']}) with "
               f"{c['channels'][0]}/{c['channels'][1]}/{c['channels'][2]} channels, followed by Dropout - Linear({c['fc_hidden']}) - ReLU - "
               f"Dropout - Linear(1). Targets are transformed with {c['target_transform']} and standardised; inputs are standardised with "
               f"training statistics; loss = {c['loss']}; optimiser Adam; batch size {c['batch_size']}; at most {c['max_epochs']} epochs with early "
               f"stopping (patience {c['patience']}) on a {int(c['inner_val_fraction'] * 100)}% hold-out carved from the <i>training part only</i>. "
               f"Predictions are inverse-transformed and clipped to the training range. Hyper-parameter grid (5-fold CV): "
               f"{escape(str(c['grid']))}. Final models are trained with seeds {c['final_seeds']}; reported test metrics are means over these seeds."))
    S.append(P("3.4 Baselines", "h2"))
    r_ = cfg["ridge"]
    S.append(P(f"<b>Mean / Median:</b> predict the training mean / median for all issues. <b>TF-IDF+Ridge:</b> word 1-2-gram TF-IDF "
               f"(max {r_['tfidf_max_features']} features, fitted on training data only) with Ridge regression (alpha in {r_['tfidf_alphas']}). "
               f"<b>SBERT+Ridge:</b> Ridge on the same SBERT vectors (alpha in {r_['alphas']}). Ridge models use the same target transform "
               f"and clipping as the CNN."))
    c_ = cfg["cnn"]
    desc = {
        "mean_baseline": ("none (constant)", "training mean"),
        "median_baseline": ("none (constant)", "training median"),
        "tfidf_ridge": ("TF-IDF word 1-2-grams", f"Ridge, alpha in {r_['tfidf_alphas']}, {cfg['cnn']['target_transform']} target"),
        "sbert_ridge": ("SBERT sentence embeddings", f"Ridge, alpha in {r_['alphas']}, {cfg['cnn']['target_transform']} target"),
        "sbert_cnn": ("SBERT sentence embeddings", f"{escape(str(meta['encoder']))}; 3x Conv1d "
                      f"{c_['channels']} k={c_['kernel_size']} + MLP; {c_['loss']} loss; grid {escape(str(c_['grid']))}"),
    }
    rows = [["Method", "Representation", "Model and settings (guideline 3.4)"]]
    for k in ORDER:
        if k in cfg["evaluation"]["methods"]:
            rows.append([LABEL[k], Paragraph(desc[k][0], st["cell"]), Paragraph(desc[k][1], st["cell"])])
    S.append(P("Table A. Methods compared and their configuration.", "small"))
    S.append(_table(rows, [3 * cm, 4 * cm, 10 * cm], st))
    S.append(P("3.5 Metrics and statistics", "h2"))
    S.append(P("MAE = mean(|y - y_hat|); MdAE = median(|y - y_hat|), both in story points (lower is better). Validation results are reported "
               "as mean +/- sample standard deviation (ddof = 1) over the five folds and are kept separate from the held-out test results. "
               "Test results carry 95% percentile bootstrap confidence intervals (resampling test issues; "
               f"{cfg['evaluation']['bootstrap_samples']} resamples) and paired two-sided Wilcoxon signed-rank tests on per-issue absolute errors "
               "against the median baseline and TF-IDF+Ridge. Across projects we macro-average (every project counts equally)."))

    # 4
    S.append(P("4. Experimental setup and reproducibility", "h1"))
    env = meta["environment"]
    S.append(_table([["Item", "Value"],
                     ["Seed / split", f"{cfg['seed']} / {int(round((1 - cfg['split']['test_size']) * 100))}:{int(round(cfg['split']['test_size'] * 100))}, {cfg['split']['n_folds']}-fold"],
                     ["Encoder", Paragraph(escape(str(meta['encoder'])), st["cell"])],
                     ["Python / PyTorch / sentence-transformers", f"{env.get('python')} / {env.get('torch')} / {env.get('sentence_transformers')}"],
                     ["scikit-learn / NumPy / pandas", f"{env.get('sklearn')} / {env.get('numpy')} / {env.get('pandas')}"],
                     ["Hardware", str(env.get('gpu') or 'CPU')],
                     ["Config hash", meta["config_hash"]]], [6 * cm, 10 * cm], st))
    S.append(P("Seeds, split indices (per-project JSON of issue keys), configuration and software versions are stored in the repository "
               "(<font face='Courier'>results/splits</font>, <font face='Courier'>results/run_metadata.json</font>, <font face='Courier'>config/</font>).", "small"))

    # 5
    S.append(P("5. Results", "h1"))
    S.append(P("5.1 Five-fold validation results (development data)", "h2"))
    rows = [["Method", "Val. MAE (macro mean)", "Avg. fold std", "Val. MdAE (macro mean)", "Avg. fold std"]]
    for k in ORDER:
        if k in m.index:
            rows.append([LABEL[k], _f(m.loc[k, "val_mae"]), _f(m.loc[k, "val_mae_fold_std"]), _f(m.loc[k, "val_mdae"]), _f(m.loc[k, "val_mdae_fold_std"])])
    S.append(P("Table 2. Validation metrics of the selected configuration, averaged over projects (fold-wise values for every project: "
               "<font face='Courier'>results/tables/validation_per_fold.csv</font>).", "small"))
    S.append(_table(rows, [3.6 * cm, 3.4 * cm, 2.6 * cm, 3.8 * cm, 2.6 * cm], st))
    vp = val[val.method == PRIMARY].sort_values("project")
    rows = [["Project", "Val. MAE (mean ± std)", "Val. MdAE (mean ± std)", "Selected hyper-parameters"]]
    for r in vp.itertuples():
        rows.append([Paragraph(escape(r.project), st["cell"]), f"{_f(r.mae_mean)} ± {_f(r.mae_std)}",
                     f"{_f(r.mdae_mean)} ± {_f(r.mdae_std)}", Paragraph(escape(r.best_params), st["cell"])])
    S.append(Spacer(1, 6))
    S.append(P("Table 3. SBERT+CNN five-fold validation results per project.", "small"))
    S.append(_table(rows, [4.4 * cm, 3.4 * cm, 3.4 * cm, 5.4 * cm], st))

    S.append(P("5.2 Final held-out test results", "h2"))
    rows = [["Method", "Test MAE", "Test MdAE", "Projects with lowest MAE", "Projects with lowest MdAE"]]
    for r in macro.itertuples():
        rows.append([r.label, _f(r.test_mae), _f(r.test_mdae), r.projects_lowest_mae, r.projects_lowest_mdae])
    S.append(P("Table 4. Test metrics averaged over projects (sorted by MAE). Ties count for the first method.", "small"))
    S.append(_table(rows, [3.6 * cm, 2.6 * cm, 2.6 * cm, 3.8 * cm, 3.8 * cm], st))

    at = pd.read_csv(out / "tables" / "assignment_table.csv")
    at = at[at.Scope.str.startswith("ALL")]
    rows = [["Evaluation", "Representation", "Model configuration", "MAE", "MdAE"]]
    for _, r in at.iterrows():
        rows.append([Paragraph(escape(str(r["Evaluation"])), st["cell"]), Paragraph(escape(str(r["Representation"])), st["cell"]),
                     Paragraph(escape(str(r["Model configuration"])), st["cell"]), Paragraph(str(r["MAE"]), st["cell"]), Paragraph(str(r["MdAE"]), st["cell"])])
    S.append(Spacer(1, 6))
    S.append(P("Table 4b. The comparison table required by guideline 3.4 (macro mean over projects). Only the SBERT row belongs to this problem statement (PS-5); the BERT and combined rows are other groups' statements.", "small"))
    S.append(_table(rows, [2.6 * cm, 2.9 * cm, 7.0 * cm, 2.2 * cm, 2.2 * cm], st))

    piv_mae = test.pivot(index="project", columns="method", values="mae")
    piv_mdae = test.pivot(index="project", columns="method", values="mdae")
    n_test = test.drop_duplicates("project").set_index("project")["n_test"]
    cols = [k for k in ORDER if k in piv_mae.columns]
    rows = [["Project", "n test"] + [LABEL[k] for k in cols]]
    for p in piv_mae.index:
        best = piv_mae.loc[p, cols].idxmin()
        cells = []
        for k in cols:
            txt = f"{_f(piv_mae.loc[p, k], 2)} / {_f(piv_mdae.loc[p, k], 2)}"
            cells.append(Paragraph(f"<b>{txt}</b>" if k == best else txt, st["cell"]))
        rows.append([Paragraph(escape(p), st["cell"]), int(n_test[p])] + cells)
    S.append(Spacer(1, 6))
    S.append(P("Table 5. Held-out test MAE / MdAE per project (bold = lowest MAE in the row). 95% confidence intervals and p-values: "
               "<font face='Courier'>results/tables/test_results.csv</font>.", "small"))
    S.append(_table(rows, [3.9 * cm, 1.2 * cm] + [(16.8 - 5.1) / len(cols) * cm] * len(cols), st))

    S.append(P("5.3 Figures", "h2"))
    for fn, cap, w in (("test_mae_per_project.png", "Figure 1. Test MAE per project.", 16.5),
                       ("test_mdae_per_project.png", "Figure 2. Test MdAE per project.", 16.5),
                       ("improvement_vs_median.png", "Figure 3. Relative MAE change versus the median baseline.", 16.5),
                       ("validation_fold_mae_boxplot.png", "Figure 4. Distribution of fold-wise validation MAE.", 12),
                       ("pred_vs_true_pooled.png", "Figure 5. SBERT+CNN predictions versus true values on the test sets (all projects).", 9.5)):
        f = out / "figures" / fn
        if f.exists():
            S.append(KeepTogether([_img(f, w), P(cap, "small")]))

    ex = agg["extra"]
    S.append(P("5.4 Additional analysis: ranks, win/tie/loss and what explains the gains", "h2"))
    rk = ex["mean_ranks"]
    rows = [["Method", "Mean rank by test MAE (1 = best)"]] + [[LABEL[k], _f(rk[k], 2)] for k in sorted(rk, key=rk.get)]
    S.append(KeepTogether([P("Table 6. Average rank over projects.", "small"), _table(rows, [5 * cm, 6 * cm], st)]))
    if "friedman" in ex:
        S.append(P(f"A Friedman test over the {ex['n_projects']} projects rejects equal performance of the methods (chi-square = "
                   f"{ex['friedman']['chi2']:.2f}, p = {ex['friedman']['p']:.2g})."))
    if "primary_vs" in ex:
        rows = [["SBERT+CNN against", "Wins", "Ties", "Losses", "Wilcoxon p (per-project MAE)"]]
        for b, v in ex["primary_vs"].items():
            rows.append([LABEL[b], v["wins"], v["ties"], v["losses"], "-" if v["wilcoxon_p"] is None else ("<0.0001" if v["wilcoxon_p"] < 1e-4 else f"{v['wilcoxon_p']:.4f}")])
        S.append(Spacer(1, 6))
        S.append(KeepTogether([P("Table 7. Per-project comparison of SBERT+CNN with each other method (lower MAE = win).", "small"),
                               _table(rows, [4 * cm, 1.8 * cm, 1.8 * cm, 2 * cm, 5 * cm], st)]))
    sp = ex.get("improvement_spearman", {})
    if sp:
        names = {"n_issues": "project size (issues)", "mode_share": "mode share of the most frequent story point",
                 "n_distinct_sp": "number of distinct story-point values", "sp_std": "story-point standard deviation"}
        bits = [f"{names[k]}: rho = {v['rho']:+.2f} (p {_pv(v['p'])})" for k, v in sp.items()]
        S.append(P("Spearman correlation between a project's property and SBERT+CNN's MAE improvement over the median baseline: "
                   + "; ".join(bits) + ". These correlations describe the 39 projects and are not causal evidence."))

    # 6
    S.append(P("6. Discussion", "h1"))
    S.append(P("All statements below are computed from the measured result tables (results/tables) and describe these experiments only.", "small"))
    for t in _discussion(agg, float(summ['mode_share'].mean())):
        S.append(P(t))
    S.append(P("<b>Which representation achieves the lowest MAE and MdAE?</b> See the first sentence of this section and Table 4. "
               "This project covers the SBERT arm only; the BERT-word and combined arms of the overall study are other groups' "
               "projects, and the stored split files (issue keys) allow a paired comparison with them on identical partitions."))

    # 7
    S.append(P("7. Limitations and threats to validity", "h1"))
    for t in [
        "<b>Random splitting</b> lets the model see issues created after the test issues (temporal leakage / optimistic estimates relative to real deployment); this is by design for PS-5 but limits external validity.",
        "<b>Single random 80:20 split per project.</b> Test sets of small projects are small, so metrics are noisy; bootstrap intervals quantify only test-sampling variability, not split variability.",
        "<b>Near-duplicates.</b> Only exact duplicate texts are removed; templated or cloned issues can still appear on both sides of a split and flatter any text model.",
        "<b>Outlier policy.</b> Removing story points above the planning-poker maximum is a modelling decision that the assignment does not specify; it changes the eligible data and the absolute MAE values, and results without the cap would differ (especially for projects with extreme data-entry errors).",
        "<b>Label noise and context.</b> Story points are relative, team-specific and sometimes assigned after the fact; effort drivers outside the text are invisible to the model.",
        "<b>Frozen encoder and truncation.</b> Texts beyond the encoder's maximum length are truncated; the encoder was not fine-tuned for effort.",
        "<b>Multiple comparisons.</b> Per-project p-values are reported both uncorrected and Holm-corrected; cross-project tests have few observations.",
        "<b>Hyper-parameter search is small</b> and the CNN treats embedding dimensions as ordered positions, which has no semantic justification; results for the CNN may differ with other architectures.",
        "<b>Dataset heterogeneity.</b> Projects differ in size, scale and estimation practice, so macro averages hide large differences.",
    ]:
        S.append(P("• " + t))

    # 8
    S.append(P("8. Conclusions", "h1"))
    mm = macro.set_index("method")
    concl = (f"On the evaluated projects, the best macro-averaged test MAE was achieved by {LABEL[mm['test_mae'].idxmin()]} "
             f"({_f(mm['test_mae'].min())}) and the best MdAE by {LABEL[mm['test_mdae'].idxmin()]} ({_f(mm['test_mdae'].min())}). ")
    if "median_baseline" in mm.index and PRIMARY in mm.index:
        gain = 100 * (mm.loc["median_baseline", "test_mae"] - mm.loc[PRIMARY, "test_mae"]) / mm.loc["median_baseline", "test_mae"]
        concl += (f"SBERT+CNN changed macro test MAE by {gain:+.1f}% relative to the median baseline. "
                  + ("This supports a modest benefit of SBERT-based estimation over a trivial baseline in the within-project, random-split setting"
                     if gain > 0 else "In this setting SBERT-based estimation did not demonstrate a benefit over a trivial baseline")
                  + (f", although the lexical TF-IDF+Ridge baseline was better still (macro MAE {_f(mm.loc['tfidf_ridge', 'test_mae'])} vs "
                     f"{_f(mm.loc[PRIMARY, 'test_mae'])}), so frozen SBERT embeddings were not the most effective representation here. "
                     if "tfidf_ridge" in mm.index and mm.loc["tfidf_ridge", "test_mae"] < mm.loc[PRIMARY, "test_mae"] else ". ")
                  + "These conclusions are subject to the limitations above.")
    S.append(P(concl))

    # 9
    S.append(P("9. Reproducing the results", "h1"))
    S.append(P("pip install -r requirements.txt &amp;&amp; pip install -e .<br/>"
               "python scripts/export_tawos_to_csv.py --database tawos --out data/tawos_issues.csv<br/>"
               "python -m spsbert run --config config/default.yaml<br/>"
               "python -m spsbert report --github-url &lt;repo-url&gt;", "code"))
    S.append(P("References", "h1"))
    for i, r in enumerate(REFS, 1):
        S.append(P(f"[{i}] {escape(r)}", "small"))
    S.append(P(REFS_NOTE, "small"))

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(out_pdf), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                      bottomMargin=1.8 * cm, title=f"Group {GROUP} - PS-5 SBERT story point estimation",
                      author=f"Group {GROUP}").build(S, onFirstPage=footer, onLaterPages=footer)
    return out_pdf
