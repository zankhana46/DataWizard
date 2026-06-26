import io
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import store
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in threads
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd


@tool
def list_datasets() -> str:
    """
    Returns the names of all CSV files available in this session
    (both user-uploaded files and pre-loaded sample datasets).
    Call this first whenever the user asks about their data.
    """
    files = store.all_files()
    if not files:
        return "No CSV files are available. Ask the user to upload a CSV file using the sidebar."
    lines = [f"- {name}: {df.shape[0]} rows × {df.shape[1]} columns" for name, df in files.items()]
    return "Available datasets:\n" + "\n".join(lines)


@tool
def load_and_cache_dataset(filename: str) -> str:
    """
    Loads a CSV dataset and returns a preview: column names, dtypes, shape, first 5 rows.
    Always call this before analyze_dataset or train_model to confirm the file exists
    and to see the exact column names.

    Args:
        filename: The exact filename as returned by list_datasets.
    """
    try:
        df = store.get(filename)
    except KeyError as e:
        return str(e)

    buf = io.StringIO()
    df.dtypes.to_string(buf)
    preview = df.head(5).to_string(index=False)
    return (
        f"Dataset: {filename}\n"
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
        f"Column dtypes:\n{buf.getvalue()}\n\n"
        f"First 5 rows:\n{preview}"
    )


@tool
def analyze_dataset(filename: str) -> str:
    """
    Runs descriptive statistics on a CSV dataset: summary stats, null counts, and insights.
    Call load_and_cache_dataset first to confirm the file exists.

    Args:
        filename: The exact filename as returned by list_datasets.
    """
    try:
        df = store.get(filename)
    except KeyError as e:
        return str(e)

    store.queue_code("analyze_dataset — equivalent code", f"""\
import pandas as pd

df = pd.read_csv("{filename}")

# Descriptive statistics
print(df.describe(include="all"))

# Missing values
print(df.isnull().sum())

# Value counts for categorical columns
for col in df.select_dtypes(exclude="number").columns:
    print(df[col].value_counts().head(3))
""")

    desc_df = df.describe(include="all").T.reset_index().rename(columns={"index": "column"})
    store.queue_dataframe(f"Descriptive statistics — {filename}", desc_df)

    desc = df.describe(include="all").to_string()
    null_counts = df.isnull().sum()
    nulls_str = null_counts[null_counts > 0].to_string() if null_counts.any() else "No missing values."

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    insights = []
    if numeric_cols:
        insights.append(f"Numeric columns: {', '.join(numeric_cols)}")
    if cat_cols:
        insights.append(f"Categorical columns: {', '.join(cat_cols)}")
    for col in cat_cols[:3]:
        top = df[col].value_counts().head(3).to_dict()
        insights.append(f"  '{col}' top values: {top}")

    return (
        f"=== Analysis: {filename} ===\n\n"
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
        f"--- Descriptive Statistics ---\n{desc}\n\n"
        f"--- Missing Values ---\n{nulls_str}\n\n"
        f"--- Insights ---\n" + "\n".join(insights)
    )


@tool
def train_model(
    filename: str,
    target_column: str,
    model_type: str,
    n_estimators: int = 100,
    test_size: float = 0.2,
    max_depth: int = 0,
) -> str:
    """
    Trains a RandomForest model on a CSV dataset and returns metrics and feature importances.
    Call load_and_cache_dataset first to confirm the file and column names.

    Args:
        filename: The exact filename as returned by list_datasets.
        target_column: The name of the column to predict.
        model_type: Either 'classification' or 'regression'.
        n_estimators: Number of trees in the forest (default 100). User can request e.g. 200 trees.
        test_size: Fraction of data held out for testing, 0.1–0.4 (default 0.2 = 20%).
        max_depth: Maximum depth of each tree (default 0 = unlimited). Set e.g. 5 to limit depth.
    """
    try:
        df = store.get(filename)
    except KeyError as e:
        return str(e)

    if target_column not in df.columns:
        return f"Column '{target_column}' not found. Available columns: {df.columns.tolist()}"

    model_type = model_type.lower().strip()
    if model_type not in ("classification", "regression"):
        return "model_type must be 'classification' or 'regression'."

    # Clamp inputs to safe ranges
    n_estimators = max(10, min(int(n_estimators), 1000))
    test_size = max(0.1, min(float(test_size), 0.4))
    max_depth_val = None if max_depth <= 0 else int(max_depth)

    df_model = df.copy()
    for col in df_model.select_dtypes(exclude="number").columns:
        if col != target_column:
            df_model = df_model.drop(columns=[col])
    df_model = df_model.dropna()

    if df_model.shape[0] < 10:
        return "Not enough rows (need at least 10 after dropping nulls) to train a model."

    X = df_model.drop(columns=[target_column])
    y = df_model[target_column]

    if X.empty:
        return "No numeric feature columns remain after removing the target."

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if model_type == "classification":
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth_val, random_state=42
        )
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
        metric_label = "Accuracy"
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth_val, random_state=42
        )
        model.fit(X_train, y_train)
        score = r2_score(y_test, model.predict(X_test))
        metric_label = "R² Score"

    importances = pd.Series(model.feature_importances_, index=X.columns)
    fi_str = importances.sort_values(ascending=False).head(10).to_string()

    if model_type == "classification":
        interp = "Excellent." if score >= 0.9 else "Good." if score >= 0.75 else "Moderate — consider more data or feature engineering."
    else:
        interp = "Explains most variance." if score >= 0.8 else "Moderate fit." if score >= 0.5 else "Low predictive power — consider adding features."

    config_used = (
        f"Config: {n_estimators} trees | "
        f"test split {int(test_size*100)}% | "
        f"max depth {'unlimited' if max_depth_val is None else max_depth_val}"
    )

    clf_class = "RandomForestClassifier" if model_type == "classification" else "RandomForestRegressor"
    metric_code = "accuracy_score(y_test, model.predict(X_test))" if model_type == "classification" \
        else "r2_score(y_test, model.predict(X_test))"
    store.queue_code("train_model — equivalent code", f"""\
import pandas as pd
from sklearn.ensemble import {clf_class}
from sklearn.model_selection import train_test_split
from sklearn.metrics import {"accuracy_score" if model_type == "classification" else "r2_score"}

df = pd.read_csv("{filename}").dropna()

# Drop non-numeric columns except target
X = df.select_dtypes(include="number").drop(columns=["{target_column}"])
y = df["{target_column}"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state=42
)

model = {clf_class}(
    n_estimators={n_estimators},
    max_depth={repr(max_depth_val)},
    random_state=42,
)
model.fit(X_train, y_train)

score = {metric_code}
print(f"Score: {{score:.4f}}")

importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False))
""")

    return (
        f"=== Model Training: {model_type.capitalize()} on '{filename}' ===\n\n"
        f"Target: {target_column} | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n"
        f"{config_used}\n"
        f"Features: {X.columns.tolist()}\n\n"
        f"--- {metric_label} ---\n{score:.4f} — {interp}\n\n"
        f"--- Top Feature Importances ---\n{fi_str}"
    )


@tool
def plot_dataset(filename: str, plot_type: str, x_column: str, y_column: str = "") -> str:
    """
    Generates a chart from a CSV dataset and queues it for display.
    Supported plot_type values:
      - "histogram"         : distribution of x_column
      - "scatter"           : x_column vs y_column
      - "bar"               : mean of y_column grouped by x_column (categorical x)
      - "boxplot"           : distribution of y_column grouped by x_column
      - "correlation"       : heatmap of all numeric columns (ignores x_column/y_column)
    Call load_and_cache_dataset first to confirm column names.

    Args:
        filename: dataset name from list_datasets
        plot_type: one of histogram, scatter, bar, boxplot, correlation
        x_column: primary column (x-axis or grouping column)
        y_column: secondary column (y-axis); not needed for histogram or correlation
    """
    try:
        df = store.get(filename)
    except KeyError as e:
        return str(e)

    if x_column and x_column not in df.columns:
        return f"Column '{x_column}' not found. Available: {df.columns.tolist()}"
    if y_column and y_column not in df.columns:
        return f"Column '{y_column}' not found. Available: {df.columns.tolist()}"

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_type = plot_type.lower().strip()
    title = ""

    try:
        if plot_type == "histogram":
            sns.histplot(df[x_column].dropna(), kde=True, ax=ax, color="steelblue")
            ax.set_xlabel(x_column)
            title = f"Distribution of {x_column}"

        elif plot_type == "scatter":
            if not y_column:
                return "scatter requires y_column."
            sns.scatterplot(data=df, x=x_column, y=y_column, ax=ax, alpha=0.6)
            title = f"{x_column} vs {y_column}"

        elif plot_type == "bar":
            if not y_column:
                return "bar requires y_column."
            grouped = df.groupby(x_column)[y_column].mean().sort_values(ascending=False)
            grouped.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
            ax.set_ylabel(f"Mean {y_column}")
            ax.set_xlabel(x_column)
            plt.xticks(rotation=45, ha="right")
            title = f"Mean {y_column} by {x_column}"

        elif plot_type == "boxplot":
            if not y_column:
                return "boxplot requires y_column."
            sns.boxplot(data=df, x=x_column, y=y_column, ax=ax)
            plt.xticks(rotation=45, ha="right")
            title = f"{y_column} by {x_column}"

        elif plot_type == "correlation":
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.shape[1] < 2:
                return "Need at least 2 numeric columns for a correlation heatmap."
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                        ax=ax, linewidths=0.5)
            title = f"Correlation Heatmap — {filename}"

        else:
            plt.close(fig)
            return f"Unknown plot_type '{plot_type}'. Use: histogram, scatter, bar, boxplot, correlation."

        ax.set_title(title)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        buf.seek(0)
        store.queue_plot(title, buf)

        # Generate equivalent matplotlib/seaborn code
        plot_code_map = {
            "histogram": f"""\
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("{filename}")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["{x_column}"].dropna(), kde=True, ax=ax, color="steelblue")
ax.set_title("Distribution of {x_column}")
plt.tight_layout()
plt.show()
""",
            "scatter": f"""\
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("{filename}")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="{x_column}", y="{y_column}", ax=ax, alpha=0.6)
ax.set_title("{x_column} vs {y_column}")
plt.tight_layout()
plt.show()
""",
            "bar": f"""\
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("{filename}")
grouped = df.groupby("{x_column}")["{y_column}"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
grouped.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Mean {y_column} by {x_column}")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
""",
            "boxplot": f"""\
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("{filename}")
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="{x_column}", y="{y_column}", ax=ax)
ax.set_title("{y_column} by {x_column}")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
""",
            "correlation": f"""\
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("{filename}")
numeric_df = df.select_dtypes(include="number")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
ax.set_title("Correlation Heatmap — {filename}")
plt.tight_layout()
plt.show()
""",
        }
        if plot_type in plot_code_map:
            store.queue_code(f"plot_dataset ({plot_type}) — equivalent code", plot_code_map[plot_type])

        plt.close(fig)
        return f"Plot '{title}' created and queued for display."

    except Exception as e:
        plt.close(fig)
        return f"Plot failed: {e}"
