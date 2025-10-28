
"""
regmodels - Advanced Regression and Visualization Library
----------------------------------------------------------
Provides easy training, evaluation, and visualization for multiple regression models.
Includes rich statistical summaries, subplot visualizations, and customizable user labels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class RegressionReport:
    def __init__(self, user_name=None, label_position="bottom", theme="light"):
        self.models = {}
        self.results = {}
        self.user_name = user_name or "Anonymous Developer"
        self.label_position = label_position
        self.theme = theme

    def add_models(self, model_names):
        model_map = {
            "linear": LinearRegression(),
            "ridge": Ridge(),
            "lasso": Lasso(),
            "decisiontree": DecisionTreeRegressor(),
            "randomforest": RandomForestRegressor(),
            "gradientboost": GradientBoostingRegressor()
        }
        for name in model_names:
            key = name.lower()
            if key in model_map:
                self.models[key] = model_map[key]
            else:
                print(f"⚠️ Unknown model '{name}', skipped.")

    def train(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            self.results[name] = {
                "y_test": y_test,
                "preds": preds,
                "r2": r2_score(y_test, preds),
                "mse": mean_squared_error(y_test, preds),
                "mae": mean_absolute_error(y_test, preds)
            }

    def show_combined_report(self):
        if not self.results:
            print("No trained models to show.")
            return

        plt.style.use("seaborn-v0_8-darkgrid" if self.theme == "dark" else "seaborn-v0_8-colorblind")
        n = len(self.results)
        fig, axes = plt.subplots(n, 1, figsize=(8, 4*n))
        if n == 1:
            axes = [axes]

        for ax, (name, res) in zip(axes, self.results.items()):
            ax.scatter(res["y_test"], res["preds"], alpha=0.6, label="Predicted vs Actual")
            ax.plot([res["y_test"].min(), res["y_test"].max()], [res["y_test"].min(), res["y_test"].max()], 'r--', label="Ideal Line")
            ax.set_title(f"{name.title()} Model Results", fontsize=12, color='darkblue')
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.legend()

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        if self.label_position == "bottom":
            fig.text(0.5, 0.02, f"Developed by {self.user_name}", ha="center", fontsize=10, color="gray")
        elif self.label_position == "top":
            fig.text(0.5, 0.98, f"Developed by {self.user_name}", ha="center", fontsize=10, color="gray")

        plt.show()

    def statistical_summary(self):
        if not self.results:
            print("No results available.")
            return None

        summary_data = []
        for name, res in self.results.items():
            summary_data.append({
                "Model": name.title(),
                "R2 Score": res["r2"],
                "MSE": res["mse"],
                "MAE": res["mae"]
            })

        df = pd.DataFrame(summary_data)
        print("\nModel Performance Summary:")
        print(df.to_string(index=False))
        return df

# Example usage (commented for safety)
# if __name__ == "__main__":
#     from sklearn.datasets import fetch_california_housing
#     data = fetch_california_housing()
#     X, y = data.data, data.target
#     rr = RegressionReport(user_name="Nabeel Nasir", label_position="bottom", theme="dark")
#     rr.add_models(["linear", "ridge", "lasso", "randomforest"])
#     rr.train(X, y)
#     rr.statistical_summary()
#     rr.show_combined_report()
