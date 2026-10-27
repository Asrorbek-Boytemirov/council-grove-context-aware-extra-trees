"""Reproduce the Council Grove lithofacies experiments reported in the paper.

The script compares a distance-based baseline and Extra Trees models under
two distinct deployment targets:

1. repeated stratified row-wise cross-validation; and
2. leave-one-well-out validation for prediction in an unseen well.

All learned preprocessing is fitted inside each training split.  The
contextual representation uses only unlabeled measurements from the current
well (robust within-well scaling, first differences, and centered rolling
summaries), which matches batch interpretation of a complete logged interval.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import seaborn as sns
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "facies_vectors.csv"
RESULTS = ROOT / "council_grove_results"
DATA_URL = (
    "https://raw.githubusercontent.com/seg/2016-ml-contest/"
    "master/facies_vectors.csv"
)
SEED = 2026
RAW_FEATURES = [
    "GR",
    "ILD_log10",
    "DeltaPHI",
    "PHIND",
    "PE",
    "NM_M",
    "RELPOS",
]
CONTEXT_LOGS = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]
CLASS_NAMES = {
    1: "SS",
    2: "CSiS",
    3: "FSiS",
    4: "SiSh",
    5: "MS",
    6: "WS",
    7: "D",
    8: "PS",
    9: "BS",
}


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        frame = pd.read_csv(DATA_PATH)
    else:
        frame = pd.read_csv(DATA_URL)
        frame.to_csv(DATA_PATH, index=False)
    return frame.sort_values(["Well Name", "Depth"]).reset_index(drop=True)


def contextual_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build label-free within-well context features.

    Robust z scores use each well's median and interquartile range.  Centered
    five-sample summaries correspond to a 2.5-ft window because the data are
    sampled every 0.5 ft.  These operations use measurements, never facies
    labels, from the well being interpreted.
    """
    output = frame[RAW_FEATURES].copy()
    grouped = frame.groupby("Well Name", sort=False)
    for column in CONTEXT_LOGS:
        median = grouped[column].transform("median")
        q1 = grouped[column].transform(lambda series: series.quantile(0.25))
        q3 = grouped[column].transform(lambda series: series.quantile(0.75))
        scale = (q3 - q1).replace(0, np.nan)
        output[f"{column}_RZ"] = (frame[column] - median) / scale
        output[f"{column}_D1"] = grouped[column].diff()
        output[f"{column}_R5M"] = grouped[column].transform(
            lambda series: series.rolling(5, center=True, min_periods=1).mean()
        )
        output[f"{column}_R5S"] = grouped[column].transform(
            lambda series: series.rolling(5, center=True, min_periods=2).std()
        )
    return output


def build_models() -> dict[str, tuple[pd.Index | list[str], Pipeline]]:
    return {
        "KNN raw": (
            RAW_FEATURES,
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        KNeighborsClassifier(
                            n_neighbors=15,
                            weights="distance",
                            metric="minkowski",
                            p=2,
                            algorithm="auto",
                            leaf_size=30,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
        "Extra Trees raw": (
            RAW_FEATURES,
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    (
                        "classifier",
                        ExtraTreesClassifier(
                            n_estimators=700,
                            max_features="sqrt",
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            bootstrap=False,
                            class_weight=None,
                            criterion="gini",
                            random_state=SEED,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        "Extra Trees context": (
            [],  # Replaced by all contextual columns after they are constructed.
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    (
                        "classifier",
                        ExtraTreesClassifier(
                            n_estimators=700,
                            max_features=0.30,
                            max_depth=16,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            bootstrap=False,
                            class_weight=None,
                            criterion="gini",
                            random_state=SEED,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    }


def align_probabilities(model: Pipeline, probability: np.ndarray, classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((len(probability), len(classes)), dtype=float)
    learned = np.asarray(model.named_steps["classifier"].classes_)
    for source_column, label in enumerate(learned):
        target_column = int(np.where(classes == label)[0][0])
        aligned[:, target_column] = probability[:, source_column]
    return aligned


def macro_auc(y_true: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> float:
    aucs = []
    for column, label in enumerate(classes):
        binary = (y_true == label).astype(int)
        if binary.min() == binary.max():
            continue
        aucs.append(roc_auc_score(binary, probability[:, column]))
    return float(np.mean(aucs)) if aucs else float("nan")


def metrics(y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, labels=classes, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, labels=classes, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, labels=classes, average="micro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred, labels=classes),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc_macro_ovr": macro_auc(y_true, probability, classes),
    }


def evaluate_model(
    model_name: str,
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    classes: np.ndarray,
    protocol: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    if protocol == "Repeated stratified 5-fold":
        splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
        iterator = splitter.split(features, target)
    elif protocol == "Leave-one-well-out":
        splitter = LeaveOneGroupOut()
        iterator = splitter.split(features, target, groups)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    split_rows: list[dict] = []
    prediction_rows: list[dict] = []
    started = time.perf_counter()
    for split_id, (train_index, test_index) in enumerate(iterator, start=1):
        fit_started = time.perf_counter()
        model.fit(features.iloc[train_index], target.iloc[train_index])
        fit_seconds = time.perf_counter() - fit_started
        probability = align_probabilities(
            model,
            model.predict_proba(features.iloc[test_index]),
            classes,
        )
        prediction = classes[np.argmax(probability, axis=1)]
        y_true = target.iloc[test_index].to_numpy()
        split_rows.append(
            {
                "model": model_name,
                "protocol": protocol,
                "split_id": split_id,
                "held_out_well": groups.iloc[test_index[0]] if protocol == "Leave-one-well-out" else "",
                "n_train": len(train_index),
                "n_test": len(test_index),
                "fit_seconds": fit_seconds,
                **metrics(y_true, prediction, probability, classes),
            }
        )
        for local_position, sample_index in enumerate(test_index):
            row = {
                "model": model_name,
                "protocol": protocol,
                "split_id": split_id,
                "sample_index": int(sample_index),
                "well": str(groups.iloc[sample_index]),
                "y_true": int(y_true[local_position]),
                "y_pred": int(prediction[local_position]),
            }
            row.update(
                {
                    f"p_{int(label)}": float(probability[local_position, column])
                    for column, label in enumerate(classes)
                }
            )
            prediction_rows.append(row)
    return pd.DataFrame(split_rows), pd.DataFrame(prediction_rows), time.perf_counter() - started


def aggregate_predictions(
    predictions: pd.DataFrame,
    target: pd.Series,
    classes: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    probability_columns = [f"p_{int(label)}" for label in classes]
    grouped = predictions.groupby("sample_index", sort=True)[probability_columns].mean()
    y_true = target.iloc[grouped.index.to_numpy()].to_numpy()
    probability = grouped.to_numpy()
    y_pred = classes[np.argmax(probability, axis=1)]
    aggregate = {
        "n_predictions": len(predictions),
        "n_unique_samples": len(grouped),
        **metrics(y_true, y_pred, probability, classes),
    }
    output = grouped.reset_index()
    output.insert(1, "y_true", y_true)
    output.insert(2, "y_pred", y_pred)
    return aggregate, output


def corrected_resampled_ttest(
    reference: np.ndarray,
    candidate: np.ndarray,
    test_train_ratio: float,
) -> dict:
    difference = np.asarray(candidate) - np.asarray(reference)
    n = len(difference)
    mean = float(difference.mean())
    variance = float(difference.var(ddof=1))
    standard_error = np.sqrt((1.0 / n + test_train_ratio) * variance)
    t_value = mean / standard_error if standard_error > 0 else np.nan
    p_value = 2 * stats.t.sf(abs(t_value), df=n - 1) if np.isfinite(t_value) else np.nan
    critical = stats.t.ppf(0.975, df=n - 1)
    return {
        "mean_difference": mean,
        "ci95_low": float(mean - critical * standard_error),
        "ci95_high": float(mean + critical * standard_error),
        "statistic": float(t_value),
        "df": n - 1,
        "p_two_sided": float(p_value),
        "method": "Nadeau-Bengio corrected resampled t-test",
    }


def paired_well_test(
    reference: pd.Series,
    candidate: pd.Series,
    metric_name: str,
    comparison: str,
) -> dict:
    reference, candidate = reference.align(candidate, join="inner")
    difference = candidate.to_numpy() - reference.to_numpy()
    rng = np.random.default_rng(SEED)
    bootstrap_means = np.array(
        [rng.choice(difference, size=len(difference), replace=True).mean() for _ in range(10000)]
    )
    test = stats.wilcoxon(candidate, reference, alternative="two-sided", zero_method="wilcox", method="auto")
    return {
        "comparison": comparison,
        "metric": metric_name,
        "mean_difference": float(difference.mean()),
        "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "statistic": float(test.statistic),
        "df": len(difference) - 1,
        "p_two_sided": float(test.pvalue),
        "method": "paired Wilcoxon signed-rank with percentile bootstrap CI",
    }


def make_figures(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    per_well: pd.DataFrame,
    final_lowo: pd.DataFrame,
    classes: np.ndarray,
) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    palette = sns.color_palette("colorblind")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), constrained_layout=True)
    counts = frame["Facies"].value_counts().sort_index()
    axes[0].bar([CLASS_NAMES[int(label)] for label in counts.index], counts.values, color=palette[0])
    axes[0].set_xlabel("Facies")
    axes[0].set_ylabel("Samples")
    axes[0].set_title("Class distribution")
    missing = frame[RAW_FEATURES].isna().mean().mul(100)
    axes[1].bar(missing.index, missing.values, color=palette[1])
    axes[1].set_xlabel("Predictor")
    axes[1].set_ylabel("Missing values (%)")
    axes[1].set_title("Predictor missingness")
    axes[1].tick_params(axis="x", rotation=35)
    fig.savefig(RESULTS / "figure_1_dataset_profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    plotted = summary.melt(
        id_vars=["model", "protocol"],
        value_vars=["accuracy", "f1_macro", "mcc"],
        var_name="metric",
        value_name="score",
    )
    grid = sns.catplot(
        data=plotted,
        x="model",
        y="score",
        hue="metric",
        col="protocol",
        kind="bar",
        palette="colorblind",
        height=3.4,
        aspect=1.15,
        sharey=True,
    )
    grid.set_axis_labels("", "Score")
    grid.set_titles("{col_name}")
    for axis in grid.axes.flat:
        axis.set_ylim(0.30, 0.90)
        axis.tick_params(axis="x", rotation=20)
    grid.fig.savefig(RESULTS / "figure_2_model_protocol_scores.png", dpi=300, bbox_inches="tight")
    plt.close(grid.fig)

    contextual = per_well[per_well["model"] == "Extra Trees context"]
    pivot = contextual.pivot(index="well", columns="protocol", values="accuracy").sort_values("Leave-one-well-out")
    fig, ax = plt.subplots(figsize=(7.7, 4.0), constrained_layout=True)
    for well, row in pivot.iterrows():
        ax.plot(
            [0, 1],
            [row["Repeated stratified 5-fold"], row["Leave-one-well-out"]],
            color="#A6A6A6",
            linewidth=1,
        )
        ax.scatter(0, row["Repeated stratified 5-fold"], color=palette[0], s=28)
        ax.scatter(1, row["Leave-one-well-out"], color=palette[3], s=28)
        ax.text(1.03, row["Leave-one-well-out"], well, va="center", fontsize=6.7)
    ax.set_xticks([0, 1], ["Row-wise", "Unseen well"])
    ax.set_xlim(-0.18, 1.55)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Context Extra Trees performance by well")
    fig.savefig(RESULTS / "figure_3_per_well_gap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    matrix = confusion_matrix(final_lowo["y_true"], final_lowo["y_pred"], labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(5.1, 4.3), constrained_layout=True)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=[CLASS_NAMES[int(label)] for label in classes],
        yticklabels=[CLASS_NAMES[int(label)] for label in classes],
        cbar_kws={"label": "Recall"},
        ax=ax,
    )
    ax.set_xlabel("Predicted facies")
    ax.set_ylabel("True facies")
    ax.set_title("Context Extra Trees leave-one-well-out")
    fig.savefig(RESULTS / "figure_4_lowo_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    frame = load_data()
    target = frame["Facies"].astype(int)
    groups = frame["Well Name"].astype(str)
    classes = np.sort(target.unique())
    raw = frame[RAW_FEATURES]
    context = contextual_features(frame)
    models = build_models()

    summary_rows: list[dict] = []
    split_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    aggregate_predictions_by_key: dict[tuple[str, str], pd.DataFrame] = {}

    for model_name, (columns, model) in models.items():
        features = context if model_name == "Extra Trees context" else raw[list(columns)]
        for protocol in ["Repeated stratified 5-fold", "Leave-one-well-out"]:
            print(f"Running {model_name} with {protocol}...", flush=True)
            split_metrics, predictions, elapsed = evaluate_model(
                model_name,
                model,
                features,
                target,
                groups,
                classes,
                protocol,
            )
            aggregate, aggregate_predictions_frame = aggregate_predictions(predictions, target, classes)
            summary_rows.append(
                {
                    "model": model_name,
                    "protocol": protocol,
                    "fits": len(split_metrics),
                    "elapsed_seconds": elapsed,
                    **aggregate,
                }
            )
            split_frames.append(split_metrics)
            prediction_frames.append(predictions)
            aggregate_predictions_by_key[(model_name, protocol)] = aggregate_predictions_frame

    summary = pd.DataFrame(summary_rows)
    split_metrics = pd.concat(split_frames, ignore_index=True)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary.to_csv(RESULTS / "model_protocol_summary.csv", index=False)
    split_metrics.to_csv(RESULTS / "split_metrics.csv", index=False)
    all_predictions.to_csv(RESULTS / "all_out_of_sample_predictions.csv", index=False)

    per_well_rows = []
    for (model_name, protocol), predictions in aggregate_predictions_by_key.items():
        indexed = predictions.set_index("sample_index")
        for well, sample_indices in frame.groupby("Well Name").groups.items():
            part = indexed.loc[np.asarray(sorted(sample_indices), dtype=int)]
            per_well_rows.append(
                {
                    "model": model_name,
                    "protocol": protocol,
                    "well": well,
                    "n": len(part),
                    "accuracy": accuracy_score(part["y_true"], part["y_pred"]),
                    "f1_macro": f1_score(
                        part["y_true"],
                        part["y_pred"],
                        labels=classes,
                        average="macro",
                        zero_division=0,
                    ),
                    "mcc": matthews_corrcoef(part["y_true"], part["y_pred"]),
                }
            )
    per_well = pd.DataFrame(per_well_rows)
    per_well.to_csv(RESULTS / "per_well_metrics.csv", index=False)

    tests: list[dict] = []
    rowwise = split_metrics[split_metrics["protocol"] == "Repeated stratified 5-fold"]
    knn = rowwise[rowwise["model"] == "KNN raw"].sort_values("split_id")
    contextual = rowwise[rowwise["model"] == "Extra Trees context"].sort_values("split_id")
    for metric_name in ["accuracy", "f1_macro", "mcc"]:
        tests.append(
            {
                "comparison": "Extra Trees context minus KNN raw under repeated row-wise validation",
                "metric": metric_name,
                **corrected_resampled_ttest(
                    knn[metric_name].to_numpy(),
                    contextual[metric_name].to_numpy(),
                    test_train_ratio=0.25,
                ),
            }
        )

    for metric_name in ["accuracy", "f1_macro"]:
        metric_pivot = per_well[
            per_well["model"] == "Extra Trees context"
        ].pivot(index="well", columns="protocol", values=metric_name)
        tests.append(
            paired_well_test(
                metric_pivot["Leave-one-well-out"],
                metric_pivot["Repeated stratified 5-fold"],
                metric_name,
                "Repeated row-wise minus leave-one-well-out for Extra Trees context",
            )
        )

        raw_context = per_well[
            per_well["protocol"] == "Leave-one-well-out"
        ].pivot(index="well", columns="model", values=metric_name)
        tests.append(
            paired_well_test(
                raw_context["Extra Trees raw"],
                raw_context["Extra Trees context"],
                metric_name,
                "Extra Trees context minus Extra Trees raw under leave-one-well-out",
            )
        )
    pd.DataFrame(tests).to_csv(RESULTS / "statistical_tests.csv", index=False)

    per_class_rows = []
    for (model_name, protocol), predictions in aggregate_predictions_by_key.items():
        for label in classes:
            per_class_rows.append(
                {
                    "model": model_name,
                    "protocol": protocol,
                    "facies": int(label),
                    "label": CLASS_NAMES[int(label)],
                    "support": int((predictions["y_true"] == label).sum()),
                    "precision": precision_score(
                        predictions["y_true"], predictions["y_pred"], labels=[label], average="macro", zero_division=0
                    ),
                    "recall": recall_score(
                        predictions["y_true"], predictions["y_pred"], labels=[label], average="macro", zero_division=0
                    ),
                    "f1": f1_score(
                        predictions["y_true"], predictions["y_pred"], labels=[label], average="macro", zero_division=0
                    ),
                    "auc_ovr": roc_auc_score(
                        (predictions["y_true"].to_numpy() == label).astype(int),
                        predictions[f"p_{int(label)}"].to_numpy(),
                    ),
                }
            )
    pd.DataFrame(per_class_rows).to_csv(RESULTS / "per_class_metrics.csv", index=False)

    pd.DataFrame(
        {
            "item": ["observations_N", "raw_features_d", "context_features_d", "classes_C", "wells", "missing_PE_n", "missing_PE_pct"],
            "value": [len(frame), len(RAW_FEATURES), context.shape[1], len(classes), groups.nunique(), int(frame["PE"].isna().sum()), frame["PE"].isna().mean() * 100],
        }
    ).to_csv(RESULTS / "dataset_summary.csv", index=False)
    target.value_counts().sort_index().rename("n_c").to_csv(RESULTS / "class_distribution.csv")
    frame.groupby("Well Name").size().sort_index().rename("n").to_csv(RESULTS / "well_distribution.csv")
    frame[RAW_FEATURES].isna().mean().mul(100).rename("missing_pct").to_csv(RESULTS / "missing_values.csv")

    final_lowo = aggregate_predictions_by_key[("Extra Trees context", "Leave-one-well-out")]
    make_figures(frame, summary, per_well, final_lowo, classes)

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "random_seed": SEED,
        "data_url": DATA_URL,
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
    }
    (RESULTS / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Results written to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
