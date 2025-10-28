"""
report_generator.py
-------------------
Drop this file into your `regmodels/` package folder.

Features:
- Interactive HTML report (plotly) with:
  - Dataset summary & correlation heatmap
  - Per-model Actual vs Predicted subplots (grid)
  - Feature importances (for estimators with `feature_importances_`)
  - Residual distributions
  - Performance bar charts (R2, RMSE, MAE)
  - Scrollable statistical summary table
  - Developer attribution (top or bottom)
  - Theming: 'light', 'dark', 'vibrant', 'professional'

- Optional static PDF output (report.pdf) using Matplotlib snapshots and ReportLab
- Single function `generate_report(...)` plus CLI support when run as __main__

Dependencies:
    pandas, numpy, scikit-learn, plotly, matplotlib, reportlab

Example:
    from regmodels.report_generator import generate_report
    generate_report(models, X_test, y_test, dataset_df=df,
                    developer_name="Your Name", name_position="top",
                    theme="professional", html_output="report.html", pdf_output="report.pdf")
"""

import os
import math
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

# Plotly for interactive HTML
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Matplotlib + ReportLab for optional PDF
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch

# Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score
)

# -------------------------
# THEME CONFIGURATION
# -------------------------
THEMES = {
    "light": {
        "bg": "#ffffff",
        "text": "#111111",
        "palette": px.colors.qualitative.Plotly
    },
    "dark": {
        "bg": "#0f0f0f",
        "text": "#f2f2f2",
        "palette": px.colors.qualitative.D3
    },
    "vibrant": {
        "bg": "#ffffff",
        "text": "#111111",
        "palette": px.colors.qualitative.Bold
    },
    "professional": {
        "bg": "#ffffff",
        "text": "#111111",
        "palette": px.colors.sequential.Teal
    }
}

# -------------------------
# METRICS HELPER
# -------------------------
def calc_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    evs = float(explained_variance_score(y_true, y_pred))
    return {"MAE": float(mae), "MSE": float(mse), "RMSE": rmse, "R2": r2, "Explained_Variance": evs}

# -------------------------
# MAIN GENERATOR FUNCTION
# -------------------------
def generate_report(
    models: Dict[str, object],
    X_test,
    y_test,
    feature_names: Optional[Sequence[str]] = None,
    dataset_df: Optional[pd.DataFrame] = None,
    developer_name: Optional[str] = None,
    name_position: str = "bottom",
    theme: str = "professional",
    html_output: str = "report.html",
    pdf_output: Optional[str] = None,
    open_in_browser: bool = False
) -> pd.DataFrame:
    """
    Generate an interactive HTML report (and optional PDF) summarizing model performance.

    Args:
        models: Dict[name -> fitted sklearn-like model] (must implement .predict)
        X_test: Test features (DataFrame or 2D array)
        y_test: Test target (1d-array or Series)
        feature_names: Optional list of feature names
        dataset_df: Optional full dataset (DataFrame) to compute dataset-level stats and correlation
        developer_name: Optional developer attribution string
        name_position: 'top' or 'bottom'
        theme: one of THEMES keys: 'light', 'dark', 'vibrant', 'professional'
        html_output: path to save interactive HTML report
        pdf_output: optional path to save static PDF report (if None, skip PDF)
        open_in_browser: if True, attempt to open HTML automatically (best-effort)
    Returns:
        pandas.DataFrame with aggregated metrics per model
    """

    # Normalize inputs
    if isinstance(X_test, pd.DataFrame):
        X_test_df = X_test.copy()
    else:
        X_test_df = pd.DataFrame(X_test, columns=feature_names) if feature_names is not None else pd.DataFrame(X_test)

    y_true = np.asarray(y_test).ravel()
    if feature_names is None and hasattr(X_test_df, "columns"):
        feature_names = list(X_test_df.columns)

    theme_cfg = THEMES.get(theme, THEMES["professional"])
    palette = theme_cfg["palette"]

    # Compute predictions + metrics
    predictions = {}
    metrics = {}
    for name, model in models.items():
        try:
            # Support pandas DataFrame or numpy arrays
            if isinstance(X_test_df, pd.DataFrame):
                y_pred = model.predict(X_test_df)
            else:
                y_pred = model.predict(X_test_df.values)
        except Exception:
            # fallback
            y_pred = model.predict(np.asarray(X_test_df))
        y_pred = np.asarray(y_pred).ravel()
        predictions[name] = y_pred
        metrics[name] = calc_metrics(y_true, y_pred)

    metrics_df = pd.DataFrame(metrics).T

    # -------------------------
    # Build interactive sections with Plotly
    # -------------------------
    html_sections = []

    # Header / Title
    header_html = f"<h1 style='text-align:center'>Regression Model Comparison Report</h1>"
    if developer_name and name_position.lower() == "top":
        header_html += f"<p style='text-align:center; font-style:italic; color:gray'>Developed by {developer_name}</p>"
    html_sections.append(header_html)

    # Dataset summary and correlation
    fig_corr = None
    if dataset_df is not None:
        try:
            stats_html = dataset_df.describe().T.round(4).to_html(classes="table table-sm", border=0)
            html_sections.append("<h2>Dataset Summary</h2>")
            html_sections.append(stats_html)

            corr = dataset_df.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                                 title="Correlation matrix")
            fig_corr.update_layout(plot_bgcolor=theme_cfg["bg"], paper_bgcolor=theme_cfg["bg"], font_color=theme_cfg["text"])
        except Exception:
            fig_corr = None

    # Feature importance (for models with attribute)
    featimp_fig = None
    fi_models = {}
    if feature_names is not None:
        for name, model in models.items():
            if hasattr(model, "feature_importances_"):
                try:
                    fi = np.array(getattr(model, "feature_importances_")).ravel()
                    fi_models[name] = fi
                except Exception:
                    pass

    if fi_models:
        rows_fi = len(fi_models)
        fig_feat = make_subplots(rows=rows_fi, cols=1, subplot_titles=list(fi_models.keys()), shared_xaxes=False)
        for idx, (mname, fi) in enumerate(fi_models.items(), start=1):
            sorted_idx = np.argsort(fi)[::-1]
            top_k = min(len(fi), 15)
            idxs = sorted_idx[:top_k]
            labels = [feature_names[i] for i in idxs]
            values = fi[idxs]
            fig_feat.add_trace(go.Bar(x=labels, y=values, name=mname, marker_color=palette[idx % len(palette)]), row=idx, col=1)
        fig_feat.update_layout(height=250*rows_fi, title_text="Feature Importances (top features)",
                               plot_bgcolor=theme_cfg["bg"], paper_bgcolor=theme_cfg["bg"], font_color=theme_cfg["text"])
        featimp_fig = fig_feat

    # Per-model Actual vs Predicted subplots
    names = list(models.keys())
    num_models = len(names)
    cols = 2
    rows = math.ceil(num_models / cols)
    fig_models = make_subplots(rows=rows, cols=cols, subplot_titles=names, horizontal_spacing=0.07, vertical_spacing=0.12)

    for idx, name in enumerate(names):
        r = idx // cols + 1
        c = idx % cols + 1
        preds = predictions[name]
        color = palette[idx % len(palette)]
        fig_models.add_trace(go.Scatter(x=y_true, y=preds, mode="markers",
                                        marker=dict(color=color, opacity=0.7, size=6),
                                        name=f"{name} preds"), row=r, col=c)
        mn = float(min(y_true.min(), preds.min()))
        mx = float(max(y_true.max(), preds.max()))
        fig_models.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                        line=dict(color="black", dash="dash"), showlegend=False), row=r, col=c)
        fig_models.update_xaxes(title_text="Actual", row=r, col=c)
        fig_models.update_yaxes(title_text="Predicted", row=r, col=c)

    fig_models.update_layout(height=350*rows, title_text="Actual vs Predicted (per model)",
                             plot_bgcolor=theme_cfg["bg"], paper_bgcolor=theme_cfg["bg"], font_color=theme_cfg["text"])

    # Performance comparison (bar charts): R2, RMSE, MAE
    perf_df = metrics_df.reset_index().rename(columns={"index": "Model"})
    fig_perf = make_subplots(rows=1, cols=3, subplot_titles=("R² Score", "RMSE (lower better)", "MAE (lower better)"))
    fig_perf.add_trace(go.Bar(x=perf_df["Model"], y=perf_df["R2"], marker_color=palette), row=1, col=1)
    fig_perf.add_trace(go.Bar(x=perf_df["Model"], y=perf_df["RMSE"], marker_color=palette), row=1, col=2)
    fig_perf.add_trace(go.Bar(x=perf_df["Model"], y=perf_df["MAE"], marker_color=palette), row=1, col=3)
    fig_perf.update_layout(height=420, showlegend=False, plot_bgcolor=theme_cfg["bg"], paper_bgcolor=theme_cfg["bg"], font_color=theme_cfg["text"])

    # Residual distributions
    fig_resid = make_subplots(rows=math.ceil(num_models / 2), cols=2, subplot_titles=names)
    for idx, name in enumerate(names):
        r = idx // 2 + 1
        c = idx % 2 + 1
        resid = y_true - predictions[name]
        fig_resid.add_trace(go.Histogram(x=resid, nbinsx=30, name=name, marker_color=palette[idx % len(palette)], opacity=0.75), row=r, col=c)
        fig_resid.update_xaxes(title_text="Residual", row=r, col=c)
        fig_resid.update_yaxes(title_text="Count", row=r, col=c)
    fig_resid.update_layout(height=300*math.ceil(num_models / 2), title_text="Residual distributions",
                            plot_bgcolor=theme_cfg["bg"], paper_bgcolor=theme_cfg["bg"], font_color=theme_cfg["text"])

    # -------------------------
    # Compose HTML document
    # -------------------------
    html = []
    html.append(f"<div style='background:{theme_cfg['bg']}; color:{theme_cfg['text']}; padding:14px;'>")
    html.append(header_html)

    if fig_corr is not None:
        html.append("<h2>Correlation Matrix</h2>")
        html.append(pio.to_html(fig_corr, include_plotlyjs=False, full_html=False, div_id="corrdiv"))

    html.append("<h2>Model Results (Actual vs Predicted)</h2>")
    html.append(pio.to_html(fig_models, include_plotlyjs=False, full_html=False, div_id="modelsdiv"))

    if featimp_fig is not None:
        html.append("<h2>Feature Importances</h2>")
        html.append(pio.to_html(featimp_fig, include_plotlyjs=False, full_html=False, div_id="featimpdiv"))

    html.append("<h2>Performance Comparison</h2>")
    html.append(pio.to_html(fig_perf, include_plotlyjs=False, full_html=False, div_id="perfdiv"))

    html.append("<h2>Residual Analysis</h2>")
    html.append(pio.to_html(fig_resid, include_plotlyjs=False, full_html=False, div_id="residdiv"))

    # Metrics table
    html.append("<h2>Statistical Summary Table</h2>")
    html.append(metrics_df.round(4).to_html(classes='table table-striped', border=0))

    if developer_name and name_position.lower() == "bottom":
        html.append(f"<p style='text-align:center; font-style:italic; color:gray'>Developed by {developer_name}</p>")

    html.append("</div>")

    # Build full HTML with plotly CDN
    html_doc = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Regression Model Comparison Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
      body {{ margin: 12px; background: {theme_cfg['bg']}; color: {theme_cfg['text']}; font-family: Arial, sans-serif; }}
      h1, h2 {{ color: {theme_cfg['text']}; }}
      .table {{ width: 100%; border-collapse: collapse; }}
      .table th, .table td {{ padding: 6px 8px; border: 1px solid #ddd; }}
    </style>
  </head>
  <body>
    {''.join(html)}
  </body>
</html>"""

    # Save HTML
    with open(html_output, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"[+] Interactive HTML report saved to: {html_output}")

    # Optional: open in browser (best-effort)
    if open_in_browser:
        try:
            import webbrowser
            webbrowser.open("file://" + os.path.realpath(html_output))
        except Exception:
            pass

    # -------------------------
    # Optional PDF generation
    # -------------------------
    if pdf_output:
        img_paths = []
        # 1) static models subplot (matplotlib)
        try:
            cols_m = 2
            rows_m = math.ceil(num_models / cols_m)
            fig_m, axes = plt.subplots(rows_m, cols_m, figsize=(12, 5 * rows_m))
            axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
            for i, name in enumerate(names):
                ax = axes_flat[i]
                ax.scatter(y_true, predictions[name], alpha=0.7, s=20)
                mn = min(y_true.min(), predictions[name].min())
                mx = max(y_true.max(), predictions[name].max())
                ax.plot([mn, mx], [mn, mx], linestyle="--", color="black")
                ax.set_title(f"{name}\nR2={metrics[name]['R2']:.3f}, RMSE={metrics[name]['RMSE']:.3f}")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
            for j in range(i+1, len(axes_flat)):
                axes_flat[j].axis("off")
            mpath = pdf_output + "_models.png"
            fig_m.tight_layout()
            fig_m.savefig(mpath, dpi=150, bbox_inches="tight")
            plt.close(fig_m)
            img_paths.append(mpath)
        except Exception as e:
            print("[!] Could not produce models image for PDF:", e)

        # 2) performance bars
        try:
            fig_p, axp = plt.subplots(figsize=(9, 4))
            x = np.arange(len(names))
            width = 0.25
            axp.bar(x - width, [metrics[n]["R2"] for n in names], width, label="R2")
            axp.bar(x, [metrics[n]["RMSE"] for n in names], width, label="RMSE")
            axp.bar(x + width, [metrics[n]["MAE"] for n in names], width, label="MAE")
            axp.set_xticks(x)
            axp.set_xticklabels(names, rotation=35, ha="right")
            axp.legend()
            fig_p.tight_layout()
            ppath = pdf_output + "_perf.png"
            fig_p.savefig(ppath, dpi=150, bbox_inches="tight")
            plt.close(fig_p)
            img_paths.append(ppath)
        except Exception as e:
            print("[!] Could not produce performance image for PDF:", e)

        # 3) correlation static
        if dataset_df is not None:
            try:
                corr = dataset_df.corr()
                fig_c, axc = plt.subplots(figsize=(7, 6))
                im = axc.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
                axc.set_xticks(range(len(corr.columns)))
                axc.set_yticks(range(len(corr.index)))
                axc.set_xticklabels(corr.columns, rotation=90)
                axc.set_yticklabels(corr.index)
                fig_c.colorbar(im, ax=axc)
                fig_c.tight_layout()
                cpath = pdf_output + "_corr.png"
                fig_c.savefig(cpath, dpi=150, bbox_inches="tight")
                plt.close(fig_c)
                img_paths.append(cpath)
            except Exception as e:
                print("[!] Could not produce correlation image for PDF:", e)

        # Build PDF via ReportLab
        try:
            doc = SimpleDocTemplate(pdf_output, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            story.append(Paragraph("Regression Model Comparison Report", styles["Title"]))
            if developer_name and name_position.lower() == "top":
                story.append(Paragraph(f"Developed by {developer_name}", styles["Normal"]))
            story.append(Spacer(1, 12))

            story.append(Paragraph("Statistical Summary", styles["Heading2"]))
            # metrics table
            table_data = [["Model"] + list(metrics_df.columns)]
            for mname in metrics_df.index:
                row = [mname] + [f"{metrics_df.loc[mname, c]:.4f}" for c in metrics_df.columns]
                table_data.append(row)
            t = Table(table_data, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            story.append(t)
            story.append(Spacer(1, 12))

            # images
            for ip in img_paths:
                try:
                    story.append(Image(ip, width=6.5 * inch, height=4 * inch))
                    story.append(Spacer(1, 12))
                except Exception:
                    pass

            if developer_name and name_position.lower() == "bottom":
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"Developed by {developer_name}", styles["Normal"]))

            doc.build(story)
            print(f"[+] Static PDF saved to: {pdf_output}")
        except Exception as e:
            print("[!] Failed to build PDF:", e)

    return metrics_df

# -------------------------
# CLI wrapper for quick execution
# -------------------------
def _demo_run_default(html_output="report.html", pdf_output="report.pdf"):
    """Quick demo using California housing dataset and three models."""
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor

    data = fetch_california_housing(as_frame=True)
    X = data.frame.drop(columns=["MedHouseVal"])
    y = data.frame["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression().fit(X_train, y_train),
        "Ridge": Ridge(alpha=1.0).fit(X_train, y_train),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    }

    generate_report(models, X_test, y_test, feature_names=list(X.columns), dataset_df=data.frame,
                    developer_name="Your Name Here", name_position="top", theme="professional",
                    html_output=html_output, pdf_output=pdf_output, open_in_browser=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate interactive regression comparison report (HTML + optional PDF).")
    parser.add_argument("--html", type=str, default="report.html", help="Path for HTML output")
    parser.add_argument("--pdf", type=str, default=None, help="Path for PDF output (optional)")
    parser.add_argument("--name", type=str, default=None, help="Developer name to show on report")
    parser.add_argument("--pos", type=str, choices=["top", "bottom"], default="bottom", help="Where to place developer name")
    parser.add_argument("--theme", type=str, choices=list(THEMES.keys()), default="professional", help="Color theme")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo using California housing dataset")
    args = parser.parse_args()

    if args.demo:
        _demo_run_default(html_output=args.html, pdf_output=args.pdf if args.pdf else "report.pdf")
    else:
        print("No demo requested. To run demo use: python report_generator.py --demo")
        print("Or import generate_report from this module and call it from your code.")
