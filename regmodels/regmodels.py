"""
regmodels.py
Professional regression model comparison library with:
✅ Subplots for multiple models
✅ Statistical analysis
✅ Scrollable HTML summary
✅ Developer credit placement
✅ Light/Dark theme support
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score
)
import pandas as pd
import math
from IPython.display import display, HTML


def train_and_evaluate(models, X_train, y_train, X_test, y_test,
                       developer_name=None, name_position="bottom",
                       theme="light"):
    """
    Train and visualize multiple regression models with detailed report.

    Args:
        models (dict): {model_name: model_instance}
        X_train, y_train, X_test, y_test: data arrays
        developer_name (str): Name to display on the report (optional)
        name_position (str): 'top' or 'bottom'
        theme (str): 'light' or 'dark'

    Returns:
        pd.DataFrame: DataFrame containing all statistical results
    """

    results = {}
    predictions = {}

    # Set theme
    if theme == "dark":
        plt.style.use("dark_background")
        scatter_colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        text_color = "white"
        bg_color = "#0f0f0f"
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
        scatter_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        text_color = "black"
        bg_color = "#ffffff"

    print("🚀 Starting Regression Model Analysis...\n")

    # Train and evaluate
    for name, model in models.items():
        print("=" * 70)
        print(f"🔹 Training Model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)

        results[name] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Explained_Variance": evs
        }

        print(f"✅ {name}")
        print(f"   MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f} | EVS: {evs:.4f}")

    # DataFrame of results
    results_df = pd.DataFrame(results).T
    print("\n📊 Statistical Summary Table:")
    print(results_df.round(4))

    # --- Visualization ---
    num_models = len(models)
    cols = 2
    rows = math.ceil(num_models / cols)

    fig = plt.figure(figsize=(12, 5 * rows), facecolor=bg_color)
    gs = gridspec.GridSpec(rows, cols, figure=fig)

    for i, (name, y_pred) in enumerate(predictions.items()):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c])
        ax.scatter(y_test, y_pred, alpha=0.7, s=40, color=scatter_colors[i])
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
        ax.set_title(f"{name}\nR²={results[name]['R2']:.3f}, RMSE={results[name]['RMSE']:.3f}",
                     fontsize=11, color=text_color)
        ax.set_xlabel("Actual Values", color=text_color)
        ax.set_ylabel("Predicted Values", color=text_color)
        ax.tick_params(colors=text_color)

    # Hide unused subplots
    for j in range(i + 1, rows * cols):
        fig.add_subplot(gs[j // cols, j % cols]).axis("off")

    # Main title
    plt.suptitle("Regression Model Comparison Report", fontsize=15, weight="bold", color=text_color, y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Add developer name *after* layout adjustment
    if developer_name:
        y_pos = 0.99 if name_position.lower() == "top" else 0.02
        fig.text(0.5, y_pos, f"Developed by {developer_name}",
                 ha='center', va='center', fontsize=11, color='gray', style='italic')

    plt.show()

    # --- Scrollable HTML Summary ---
    html_table = results_df.round(4).to_html(classes="table table-striped", border=0)
    scrollable_html = f"""
    <div style='
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #ccc;
        border-radius: 8px;
        margin-top: 15px;
        padding: 10px;
        background-color: {bg_color};
        color: {text_color};
    '>
        <h3 style='text-align:center; color:{text_color};'>📄 Statistical Summary</h3>
        {html_table}
    </div>
    """
    try:
        display(HTML(scrollable_html))  # Works in Jupyter / Colab
    except Exception:
        print("\n(💡 Tip: Scrollable HTML view available in Jupyter environments.)")

    return results_df
