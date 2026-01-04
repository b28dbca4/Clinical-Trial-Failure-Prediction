from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

# =====================================================================
# GLOBAL STYLE & CONSTANTS
# =====================================================================

sns.set(style="whitegrid", font_scale=1.1)
DEFAULT_FIGSIZE: Tuple[int, int] = (8, 5)
DEFAULT_DPI: int = 150
ALPHA_LEVEL: float = 0.05

# Color palettes for EDA
OUTCOME_COLORS: Dict[int, str] = {
    0: '#e74c3c',  # Fail - Red
    1: '#27ae60'   # Success - Green
}

SPONSOR_COLORS: Dict[str, str] = {
    'INDUSTRY': '#3498db',
    'OTHER': '#95a5a6',
    'NIH': '#9b59b6',
    'FED': '#e67e22',
    'NETWORK': '#1abc9c',
    'INDIV': '#f1c40f'
}


# =====================================================================
# DATA CLASSES FOR STATISTICAL RESULTS
# =====================================================================

@dataclass
class ChiSquareResult:
    """Kết quả kiểm định Chi-square với effect size."""
    chi2: float
    p_value: float
    dof: int
    expected: np.ndarray
    cramers_v: float
    effect_interpretation: str
    n_samples: int
    n_rows: int
    n_cols: int
    is_significant: bool
    
    def __str__(self) -> str:
        sig = "Có ý nghĩa" if self.is_significant else "Không có ý nghĩa"
        return (
            f"Chi-square: χ²={self.chi2:.4f}, p={self.p_value:.2e}, "
            f"Cramér's V={self.cramers_v:.4f} ({self.effect_interpretation}), {sig}"
        )


@dataclass
class MannWhitneyResult:
    """Kết quả kiểm định Mann-Whitney U với effect size."""
    statistic: float
    p_value: float
    rank_biserial: float
    effect_interpretation: str
    n1: int
    n2: int
    median1: float
    median2: float
    is_significant: bool
    
    def __str__(self) -> str:
        sig = "Có ý nghĩa" if self.is_significant else "Không có ý nghĩa"
        return (
            f"Mann-Whitney U: U={self.statistic:,.0f}, p={self.p_value:.2e}, "
            f"r={self.rank_biserial:.4f} ({self.effect_interpretation}), {sig}"
        )


# =====================================================================
# UTILITY
# =====================================================================

def save_fig(
    fig: plt.Figure,
    path: str | Path,
    dpi: int = DEFAULT_DPI,
    tight: bool = True,
) -> None:
    """
    Lưu figure ra file với cấu hình thống nhất.

    Args:
        fig: matplotlib Figure
        path: đường dẫn file (png, pdf, ...)
        dpi: độ phân giải
        tight: có gọi tight_layout hay không
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# EDA – LABEL & DISTRIBUTIONS
# =====================================================================

def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "label",
    ax: Optional[plt.Axes] = None,
    normalize: bool = False,
) -> plt.Axes:
    """
    Vẽ phân phối nhãn (binary hoặc multi-class).

    Args:
        df: DataFrame chứa target
        target_col: tên cột nhãn
        ax: axes để vẽ (nếu None sẽ tạo mới)
        normalize: nếu True vẽ tỉ lệ (%), ngược lại vẽ count

    Returns:
        axes đã vẽ
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    value_counts = df[target_col].value_counts(normalize=normalize).sort_index()
    value_counts.plot(kind="bar", ax=ax)

    ax.set_title(f"Distribution of {target_col}")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Percentage" if normalize else "Count")

    if normalize:
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f"{height * 100:.1f}%",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f"{int(height):,}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    return ax


def plot_numeric_hist(
    df: pd.DataFrame,
    col: str,
    bins: int = 30,
    ax: Optional[plt.Axes] = None,
    kde: bool = True,
) -> plt.Axes:
    """
    Vẽ histogram cho một biến numeric.

    Args:
        df: DataFrame
        col: tên cột numeric
        bins: số bins
        ax: axes để vẽ
        kde: có vẽ đường KDE không

    Returns:
        axes đã vẽ
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    sns.histplot(df[col].dropna(), bins=bins, kde=kde, ax=ax)
    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

    return ax


def plot_missing_rates(
    df: pd.DataFrame,
    top_n: int = 30,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Vẽ biểu đồ tỉ lệ missing cho các cột (top_n cột missing nhiều nhất).

    Args:
        df: DataFrame
        top_n: số cột hiển thị
        ax: axes

    Returns:
        axes đã vẽ
    """
    missing = df.isna().mean().sort_values(ascending=False).head(top_n)
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(8, len(missing) * 0.3), DEFAULT_FIGSIZE[1])
        )

    missing.mul(100).plot(kind="barh", ax=ax)
    ax.set_xlabel("Missing rate (%)")
    ax.set_ylabel("Columns")
    ax.set_title(f"Missing Rate (Top {top_n})")

    # thêm label %
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f"{width:.1f}%",
            (width + 0.5, p.get_y() + p.get_height() / 2),
            va="center",
            fontsize=9,
        )

    return ax


# =====================================================================
# EDA – CORRELATION & TOP CATEGORIES
# =====================================================================

def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    method: str = "pearson",
    ax: Optional[plt.Axes] = None,
    annot: bool = False,
) -> plt.Axes:
    """
    Vẽ heatmap tương quan giữa các biến numeric.

    Args:
        df: DataFrame
        cols: list cột cần xét (None = tất cả numeric)
        method: 'pearson', 'spearman', 'kendall'
        ax: axes
        annot: có in số corr trên từng ô không

    Returns:
        axes đã vẽ
    """
    if cols is None:
        corr_df = df.select_dtypes(include=[np.number])
    else:
        corr_df = df[list(cols)]

    corr = corr_df.corr(method=method)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.6), max(6, len(corr) * 0.6)))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        annot=annot,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(f"Correlation Heatmap ({method.capitalize()})")

    return ax


def plot_top_categories(
    df: pd.DataFrame,
    col: str,
    top_n: int = 10,
    horizontal: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Vẽ top_n giá trị xuất hiện nhiều nhất của một cột phân loại.

    Args:
        df: DataFrame
        col: tên cột
        top_n: số categories
        horizontal: vẽ bar ngang hay dọc
        ax: axes

    Returns:
        axes đã vẽ
    """
    vc = df[col].value_counts().head(top_n)

    if ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    if horizontal:
        vc.plot(kind="barh", ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
    else:
        vc.plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        ax.set_xlabel(col)
        ax.tick_params(axis="x", rotation=45)

    ax.set_title(f"Top {top_n} values of {col}")

    # label count
    for p in ax.patches:
        if horizontal:
            width = p.get_width()
            ax.annotate(
                f"{int(width):,}",
                (width + max(vc) * 0.01, p.get_y() + p.get_height() / 2),
                va="center",
                fontsize=9,
            )
        else:
            height = p.get_height()
            ax.annotate(
                f"{int(height):,}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    return ax


# =====================================================================
# MODEL EVALUATION PLOTS
# =====================================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    normalize: Optional[str] = None,
) -> plt.Axes:
    """
    Vẽ confusion matrix.

    Args:
        y_true: nhãn thật
        y_pred: nhãn dự đoán
        labels: tên lớp (hiển thị trên trục)
        ax: axes
        normalize: None, 'true', 'pred' hoặc 'all'

    Returns:
        axes đã vẽ
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        display_labels=labels,
        normalize=normalize,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Confusion Matrix")

    return ax


def plot_roc_curve(
    y_true,
    y_score,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Vẽ ROC curve cho bài toán binary classification.

    Args:
        y_true: nhãn thật (0/1)
        y_score: xác suất dự đoán cho lớp positive (hoặc score)

    Returns:
        axes đã vẽ
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("ROC Curve")

    return ax


def plot_pr_curve(
    y_true,
    y_score,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Vẽ Precision–Recall curve cho bài toán binary classification.

    Args:
        y_true: nhãn thật (0/1)
        y_score: xác suất dự đoán cho lớp positive (hoặc score)

    Returns:
        axes đã vẽ
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("Precision–Recall Curve")

    return ax


def plot_feature_importance(
    feature_names: Sequence[str],
    importances: Sequence[float],
    top_n: int = 20,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Vẽ feature importance (thường dùng cho tree-based models).

    Args:
        feature_names: danh sách tên feature
        importances: độ quan trọng tương ứng
        top_n: số feature hiển thị
        ax: axes

    Returns:
        axes đã vẽ
    """
    feature_names = np.array(feature_names)
    importances = np.array(importances)

    idx = np.argsort(importances)[::-1][:top_n]
    feats = feature_names[idx]
    vals = importances[idx]

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(DEFAULT_FIGSIZE[0], max(4, 0.3 * len(feats)))
        )

    ax.barh(range(len(feats)), vals[::-1])
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance (Top {top_n})")

    return ax


# =====================================================================
# PREPROCESSING / SPLIT SANITY PLOTS
# =====================================================================

def plot_missing_rate_comparison(
    missing_before: pd.Series,
    missing_after: pd.Series,
    top_n: int = 20,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    So sánh tỉ lệ missing trước và sau khi xử lý.

    Args:
        missing_before: Series index=cột, value=missing_rate trước cleaning
        missing_after: tương tự sau cleaning
        top_n: số cột hiển thị (lấy theo missing_before cao nhất)
        ax: axes

    Returns:
        axes đã vẽ
    """
    combined = pd.DataFrame({"before": missing_before, "after": missing_after}).fillna(0)
    top = combined.nlargest(top_n, "before")

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(10, len(top) * 0.4), DEFAULT_FIGSIZE[1] + 1)
        )

    x = np.arange(len(top))
    width = 0.35

    ax.barh(x - width / 2, top["before"] * 100, width, label="Before")
    ax.barh(x + width / 2, top["after"] * 100, width, label="After")

    ax.set_yticks(x)
    ax.set_yticklabels(top.index)
    ax.set_xlabel("Missing rate (%)")
    ax.set_title(f"Missing Rate Before vs After (Top {top_n})")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    return ax


def plot_split_sanity(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
    label_col: str = "label",
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Vẽ sanity check cho train/val/test:
    - Số lượng mẫu mỗi split
    - Tỉ lệ positive mỗi split

    Đồng thời in ra bảng summary để dễ đọc.

    Args:
        train_df: DataFrame train
        val_df: DataFrame val (có thể None)
        test_df: DataFrame test
        label_col: tên cột nhãn
        figsize: kích thước figure

    Returns:
        Figure chứa 2 subplot
    """
    splits = [("Train", train_df)]
    if val_df is not None and len(val_df) > 0:
        splits.append(("Validation", val_df))
    splits.append(("Test", test_df))

    summary = []
    for name, df in splits:
        n = len(df)
        if label_col in df.columns and n > 0:
            pos_rate = df[label_col].mean() * 100
            n_pos = int((df[label_col] == 1).sum())
            n_neg = int((df[label_col] == 0).sum())
        else:
            pos_rate = 0.0
            n_pos = n_neg = 0

        summary.append(
            {
                "Split": name,
                "N_samples": n,
                "Positive (1)": n_pos,
                "Negative (0)": n_neg,
                "Positive_rate_%": f"{pos_rate:.2f}",
            }
        )

    summary_df = pd.DataFrame(summary)
    print("\n=== Split sanity summary ===")
    print(summary_df.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    names = [s["Split"] for s in summary]
    sizes = [s["N_samples"] for s in summary]
    rates = [float(s["Positive_rate_%"]) for s in summary]

    # 1) size
    ax1 = axes[0]
    bars1 = ax1.bar(names, sizes)
    ax1.set_title("Split sizes")
    ax1.set_ylabel("Number of samples")
    ax1.grid(axis="y", alpha=0.3)
    for bar, size in zip(bars1, sizes):
        ax1.annotate(
            f"{size:,}",
            (bar.get_x() + bar.get_width() / 2, size),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 2) positive rate
    ax2 = axes[1]
    bars2 = ax2.bar(names, rates)
    ax2.set_title("Positive rate by split")
    ax2.set_ylabel("Positive rate (%)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(rates) * 1.3 if rates else 1)
    for bar, rate in zip(bars2, rates):
        ax2.annotate(
            f"{rate:.2f}%",
            (bar.get_x() + bar.get_width() / 2, rate),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    return fig
