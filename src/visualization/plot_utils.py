"""
Plot Utilities Module for Clinical Trial Failure Prediction

This module provides comprehensive visualization functions for:
- Data exploration and descriptive statistics
- Answering meaningful questions with clear visualizations
- Supporting model evaluation and interpretation

All visualizations are designed to be clear, informative, and publication-ready.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set default plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Default figure size and DPI for publication quality
DEFAULT_FIGSIZE = (12, 8)
DEFAULT_DPI = 100
SAVE_DPI = 300


class PlotUtils:
    """Utility class for creating various types of plots for clinical trial analysis"""
    
    def __init__(self, save_dir: Optional[str] = None, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize PlotUtils
        
        Args:
            save_dir: Directory to save plots (optional)
            style: Matplotlib style to use
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use(style)
    
    def _save_plot(self, filename: str, dpi: int = SAVE_DPI):
        """Save plot to file if save_dir is set"""
        if self.save_dir:
            filepath = self.save_dir / filename
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
    
    # ==================== DISTRIBUTION PLOTS ====================
    
    def plot_distribution(
        self,
        data: pd.Series,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        bins: int = 30,
        kde: bool = True,
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
        save_name: Optional[str] = None
    ):
        """
        Plot distribution of a single variable with histogram and optional KDE
        
        Args:
            data: Series to plot
            title: Plot title
            xlabel: X-axis label
            bins: Number of bins for histogram
            kde: Whether to show kernel density estimate
            figsize: Figure size
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram with KDE
        sns.histplot(data=data, bins=bins, kde=kde, ax=ax, color='steelblue', alpha=0.7)
        
        # Add mean and median lines
        mean_val = data.mean()
        median_val = data.median()
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        
        # Set labels and title
        ax.set_xlabel(xlabel or data.name or 'Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title or f'Distribution of {data.name or "Variable"}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            self._save_plot(save_name)
        
        plt.show()
    
    def plot_multiple_distributions(
        self,
        data: pd.DataFrame,
        columns: List[str],
        ncols: int = 3,
        figsize: Tuple[int, int] = (15, 10),
        save_name: Optional[str] = None
    ):
        """
        Plot distributions of multiple variables in a grid
        
        Args:
            data: DataFrame containing variables
            columns: List of column names to plot
            ncols: Number of columns in the grid
            figsize: Figure size
            save_name: Filename to save plot
        """
        n_plots = len(columns)
        nrows = (n_plots + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        for idx, col in enumerate(columns):
            if idx < len(axes):
                sns.histplot(data=data[col], kde=True, ax=axes[idx], color='steelblue', alpha=0.7)
                axes[idx].set_title(f'Distribution of {col}', fontsize=11, fontweight='bold')
                axes[idx].set_xlabel(col, fontsize=10)
                axes[idx].set_ylabel('Frequency', fontsize=10)
                axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_name:
            self._save_plot(save_name)
        
        plt.show()
    
    # ==================== COMPARISON PLOTS ====================
    
    def plot_categorical_comparison(
        self,
        data: pd.DataFrame,
        categorical_col: str,
        target_col: str,
        plot_type: str = 'count',
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
        save_name: Optional[str] = None
    ):
        """
        Compare target variable across categorical groups
        
        Args:
            data: DataFrame
            categorical_col: Column name for categories
            target_col: Target variable column name
            plot_type: 'count', 'proportion', or 'both'
            figsize: Figure size
            save_name: Filename to save plot
        """
        if plot_type == 'both':
            fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]))
            
            # Count plot
            sns.countplot(data=data, x=categorical_col, hue=target_col, ax=axes[0])
            axes[0].set_title(f'{target_col} Count by {categorical_col}', fontsize=12, fontweight='bold')
            axes[0].set_xlabel(categorical_col, fontsize=11)
            axes[0].set_ylabel('Count', fontsize=11)
            axes[0].legend(title=target_col)
            axes[0].tick_params(axis='x', rotation=45)
            
            # Proportion plot
            ct = pd.crosstab(data[categorical_col], data[target_col], normalize='index') * 100
            ct.plot(kind='bar', stacked=False, ax=axes[1], width=0.8)
            axes[1].set_title(f'{target_col} Proportion by {categorical_col}', fontsize=12, fontweight='bold')
            axes[1].set_xlabel(categorical_col, fontsize=11)
            axes[1].set_ylabel('Percentage (%)', fontsize=11)
            axes[1].legend(title=target_col)
            axes[1].tick_params(axis='x', rotation=45)
            
        else:
            fig, ax = plt.subplots(figsize=figsize)
            
            if plot_type == 'count':
                sns.countplot(data=data, x=categorical_col, hue=target_col, ax=ax)
                ax.set_ylabel('Count', fontsize=11)
            else:  # proportion
                ct = pd.crosstab(data[categorical_col], data[target_col], normalize='index') * 100
                ct.plot(kind='bar', stacked=False, ax=ax, width=0.8)
                ax.set_ylabel('Percentage (%)', fontsize=11)
            
            ax.set_title(f'{target_col} Distribution by {categorical_col}', fontsize=12, fontweight='bold')
            ax.set_xlabel(categorical_col, fontsize=11)
            ax.legend(title=target_col)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_name:
            self._save_plot(save_name)
        
        plt.show()
    
    def plot_boxplot_comparison(
        self,
        data: pd.DataFrame,
        categorical_col: str,
        numerical_col: str,
        hue: Optional[str] = None,
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
        save_name: Optional[str] = None
    ):
        """
        Create boxplot to compare numerical variable across categories
        
        Args:
            data: DataFrame
            categorical_col: Column name for categories (x-axis)
            numerical_col: Numerical column to compare (y-axis)
            hue: Optional column for color grouping
            figsize: Figure size
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.boxplot(data=data, x=categorical_col, y=numerical_col, hue=hue, ax=ax)
        
        ax.set_title(f'{numerical_col} by {categorical_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(categorical_col, fontsize=12)
        ax.set_ylabel(numerical_col, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name:
            self._save_plot(save_name)
        
        plt.show()
    
    def plot_violin_comparison(
        self,
        data: pd.DataFrame,
        categorical_col: str,
        numerical_col: str,
        hue: Optional[str] = None,
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
        save_name: Optional[str] = None
    ):
        """
        Create violin plot to show distribution comparison
        
        Args:
            data: DataFrame
            categorical_col: Column name for categories (x-axis)
            numerical_col: Numerical column to compare (y-axis)
            hue: Optional column for color grouping
            figsize: Figure size
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.violinplot(data=data, x=categorical_col, y=numerical_col, hue=hue, ax=ax, inner='box')
        
        ax.set_title(f'{numerical_col} Distribution by {categorical_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(categorical_col, fontsize=12)
        ax.set_ylabel(numerical_col, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name:
            self._save_plot(save_name)
        
        plt.show()
    
    # ==================== CORRELATION PLOTS ====================
    
    def plot_correlation_heatmap(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'pearson',
        figsize: Tuple[int, int] = (12, 10),
        annot: bool = True,
        cmap: str = 'coolwarm',
        save_name: Optional[str] = None
    ):
        """
        Plot correlation heatmap
        
        Args:
            data: DataFrame
            columns: Columns to include (None = all numeric)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            figsize: Figure size
            annot: Whether to annotate cells with correlation values
            cmap: Color map
            save_name: Filename to save plot
        """
        if columns:
            corr_data = data[columns]
        else:
            corr_data = data.select_dtypes(include=[np.number])
        
        corr_matrix = corr_data.corr(method=method)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=annot,
            fmt='.2f',
            cmap=cmap,
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title(f'Correlation Heatmap ({method.capitalize()})', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            self._save_plot(save_name)
        
        plt.show()
    
    def plot_correlation_with_target(
        self,
        data: pd.DataFrame,
        target_col: str,
        top_n: int = 15,
        figsize: Tuple[int, int] = (10, 8),
        save_name: Optional[str] = None
    ):
        """
        Plot correlation of features with target variable
        
        Args:
            data: DataFrame
            target_col: Target variable column name
            top_n: Number of top correlations to show
            figsize: Figure size
            save_name: Filename to save plot
        """
        # Calculate correlations with target
        correlations = data.corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
        
        # Get top N
        top_corr = correlations.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['red' if x < 0 else 'green' for x in top_corr.values]
        top_corr.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
        
        ax.set_title(f'Top {top_n} Features Correlated with {target_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Correlation Coefficient', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_name:
            self._save_plot(save_name)
        
        plt.show()
    
    # ==================== TIME SERIES PLOTS ====================
    
    def plot_time_series(
        self,
        data: pd.DataFrame,
        date_col: str,
        value_col: str,
        groupby_col: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6),
        save_name: Optional[str] = None
    ):
        """
        Plot time series data
        
        Args:
            data: DataFrame
            date_col: Column name for dates
            value_col: Column name for values to plot
            groupby_col: Optional column to group by (creates multiple lines)
            figsize: Figure size
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if groupby_col:
            for group_name, group_data in data.groupby(groupby_col):
                group_data = group_data.sort_values(date_col)
                ax.plot(group_data[date_col], group_data[value_col], marker='o', label=group_name, linewidth=2)
            ax.legend()
        else:
            data_sorted = data.sort_values(date_col)
            ax.plot(data_sorted[date_col], data_sorted[value_col], marker='o', linewidth=2, color='steelblue')
        
        ax.set_title(f'{value_col} Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel(date_col, fontsize=12)
        ax.set_ylabel(value_col, fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_name:
            self._save_plot(save_name)
        
        plt.show()
    
    def plot_trend_analysis(
        self,
        data: pd.DataFrame,
        date_col: str,
        value_col: str,
        aggregation: str = 'mean',
        freq: str = 'M',
        figsize: Tuple[int, int] = (14, 6),
        save_name: Optional[str] = None
    ):
        """
        Plot aggregated time series with trend
        
        Args:
            data: DataFrame
            date_col: Column name for dates
            value_col: Column name for values
            aggregation: Aggregation method ('mean', 'sum', 'count', 'median')
            freq: Frequency for aggregation ('D'=daily, 'W'=weekly, 'M'=monthly, 'Y'=yearly)
            figsize: Figure size
            save_name: Filename to save plot
        """
        # Ensure date column is datetime
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Set date as index and resample
        df = df.set_index(date_col)
        
        if aggregation == 'mean':
            aggregated = df[value_col].resample(freq).mean()
        elif aggregation == 'sum':
            aggregated = df[value_col].resample(freq).sum()
        elif aggregation == 'count':
            aggregated = df[value_col].resample(freq).count()
        elif aggregation == 'median':
            aggregated = df[value_col].resample(freq).median()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot aggregated data
        ax.plot(aggregated.index, aggregated.values, marker='o', linewidth=2, label='Actual', color='steelblue')
        
        # Add trend line
        x_numeric = np.arange(len(aggregated))
        z = np.polyfit(x_numeric, aggregated.values, 1)
        p = np.poly1d(z)
        ax.plot(aggregated.index, p(x_numeric), "--", linewidth=2, label='Trend', color='red', alpha=0.7)
        
        ax.set_title(f'{value_col} Trend ({aggregation.capitalize()} by {freq})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'{aggregation.capitalize()} {value_col}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_name:
            self._save_plot(save_name)
        
        plt.show()
    
    # ==================== MISSING VALUE PLOTS ====================
    
    def plot_missing_values(
        self,
        data: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 8),
        save_name: Optional[str] = None
    ):
        """
        Visualize missing values in dataset
        
        Args:
            data: DataFrame
            figsize: Figure size
            save_name: Filename to save plot
        """
        missing = data.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            print("No missing values found in the dataset!")
            return
        
        missing_percent = (missing / len(data)) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Count plot
        missing.plot(kind='barh', ax=axes[0], color='coral')
        axes[0].set_title('Missing Values Count', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Number of Missing Values', fontsize=11)
        axes[0].set_ylabel('Columns', fontsize=11)
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Percentage plot
        missing_percent.plot(kind='barh', ax=axes[1], color='steelblue')
        axes[1].set_title('Missing Values Percentage', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Percentage (%)', fontsize=11)
        axes[1].set_ylabel('Columns', fontsize=11)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_name:
            self._save_plot(save_name)
        
        plt.show()
    
    # ==================== STATISTICAL SUMMARY PLOTS ====================
    
    def plot_statistical_summary(
        self,
        data: pd.DataFrame,
        column: str,
        figsize: Tuple[int, int] = (14, 6),
        save_name: Optional[str] = None
    ):
        """
        Create comprehensive statistical summary visualization
        
        Args:
            data: DataFrame
            column: Column name to analyze
            figsize: Figure size
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Histogram with KDE
        sns.histplot(data=data[column], kde=True, ax=axes[0], color='steelblue', alpha=0.7)
        axes[0].set_title(f'Distribution of {column}', fontsize=11, fontweight='bold')
        axes[0].set_xlabel(column, fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Boxplot
        sns.boxplot(y=data[column], ax=axes[1], color='lightgreen')
        axes[1].set_title(f'Boxplot of {column}', fontsize=11, fontweight='bold')
        axes[1].set_ylabel(column, fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # QQ plot
        from scipy import stats
        stats.probplot(data[column].dropna(), dist="norm", plot=axes[2])
        axes[2].set_title(f'Q-Q Plot of {column}', fontsize=11, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""
        Mean: {data[column].mean():.2f}
        Median: {data[column].median():.2f}
        Std: {data[column].std():.2f}
        Min: {data[column].min():.2f}
        Max: {data[column].max():.2f}
        """
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_name:
            self._save_plot(save_name)
        
        plt.show()
    
    # ==================== TOP N PLOTS ====================
    
    def plot_top_n_categories(
        self,
        data: pd.DataFrame,
        column: str,
        top_n: int = 10,
        figsize: Tuple[int, int] = (12, 6),
        horizontal: bool = True,
        save_name: Optional[str] = None
    ):
        """
        Plot top N categories by frequency
        
        Args:
            data: DataFrame
            column: Column name to analyze
            top_n: Number of top categories to show
            figsize: Figure size
            horizontal: Whether to use horizontal bars
            save_name: Filename to save plot
        """
        value_counts = data[column].value_counts().head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if horizontal:
            value_counts.plot(kind='barh', ax=ax, color='steelblue', alpha=0.8)
            ax.set_xlabel('Count', fontsize=12)
            ax.set_ylabel(column, fontsize=12)
        else:
            value_counts.plot(kind='bar', ax=ax, color='steelblue', alpha=0.8)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_xlabel(column, fontsize=12)
            ax.tick_params(axis='x', rotation=45)
        
        ax.set_title(f'Top {top_n} {column}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x' if horizontal else 'y')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3)
        
        plt.tight_layout()
        
        if save_name:
            self._save_plot(save_name)
        
        plt.show()
    
    # ==================== SCATTER PLOTS ====================
    
    def plot_scatter(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        hue: Optional[str] = None,
        size: Optional[str] = None,
        add_regression: bool = False,
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
        save_name: Optional[str] = None
    ):
        """
        Create scatter plot with optional regression line
        
        Args:
            data: DataFrame
            x_col: Column for x-axis
            y_col: Column for y-axis
            hue: Column for color grouping
            size: Column for point size
            add_regression: Whether to add regression line
            figsize: Figure size
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if add_regression and not hue:
            sns.regplot(data=data, x=x_col, y=y_col, ax=ax, scatter_kws={'alpha': 0.6})
        else:
            sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue, size=size, ax=ax, alpha=0.7)
        
        # Calculate and display correlation
        corr = data[[x_col, y_col]].corr().iloc[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(f'{y_col} vs {x_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            self._save_plot(save_name)
        
        plt.show()
    
    def plot_pairplot(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        save_name: Optional[str] = None
    ):
        """
        Create pairplot for multiple variables
        
        Args:
            data: DataFrame
            columns: Columns to include (None = all numeric)
            hue: Column for color grouping
            save_name: Filename to save plot
        """
        if columns:
            plot_data = data[columns + ([hue] if hue and hue not in columns else [])]
        else:
            plot_data = data.select_dtypes(include=[np.number])
            if hue:
                plot_data[hue] = data[hue]
        
        g = sns.pairplot(plot_data, hue=hue, diag_kind='kde', corner=True)
        g.fig.suptitle('Pairwise Relationships', y=1.02, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            self._save_plot(save_name)
        
        plt.show()


# Convenience functions for quick plotting
def quick_distribution(data: pd.Series, **kwargs):
    """Quick distribution plot"""
    plotter = PlotUtils()
    plotter.plot_distribution(data, **kwargs)


def quick_correlation(data: pd.DataFrame, **kwargs):
    """Quick correlation heatmap"""
    plotter = PlotUtils()
    plotter.plot_correlation_heatmap(data, **kwargs)


def quick_comparison(data: pd.DataFrame, categorical_col: str, target_col: str, **kwargs):
    """Quick categorical comparison"""
    plotter = PlotUtils()
    plotter.plot_categorical_comparison(data, categorical_col, target_col, **kwargs)


# ============================================================================
# EDA SPECIFIC FUNCTIONS - Added for Question Analysis
# ============================================================================

def plot_phase_analysis(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Comprehensive phase analysis visualization
    
    Args:
        df: DataFrame with 'phases', 'overall_status', and 'success' columns
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter for completed/terminated/withdrawn only
    df_filtered = df[df['overall_status'].isin(['COMPLETED', 'TERMINATED', 'WITHDRAWN'])].copy()
    
    # Create success column if not exists
    if 'success' not in df_filtered.columns:
        df_filtered['success'] = df_filtered['overall_status'].apply(
            lambda x: 'Thành công' if x == 'COMPLETED' else 'Thất bại'
        )
    
    # Phase success rate
    phase_pivot = df_filtered.groupby(['phases', 'success']).size().unstack(fill_value=0)
    phase_pivot_pct = phase_pivot.div(phase_pivot.sum(axis=1), axis=0) * 100
    
    phase_pivot_pct.plot(kind='barh', stacked=True, ax=axes[0],
                          color=['#2ecc71', '#e74c3c'], width=0.7)
    axes[0].set_xlabel('Tỷ lệ (%)', fontsize=12)
    axes[0].set_ylabel('Phase', fontsize=12)
    axes[0].set_title('Tỷ Lệ Thành Công/Thất Bại Theo Phase', fontsize=14, fontweight='bold')
    axes[0].legend(title='Kết quả')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Count by phase
    phase_counts = df_filtered['phases'].value_counts().sort_index()
    axes[1].bar(range(len(phase_counts)), phase_counts.values,
                color=sns.color_palette("husl", len(phase_counts)))
    axes[1].set_xticks(range(len(phase_counts)))
    axes[1].set_xticklabels(phase_counts.index, rotation=45, ha='right')
    axes[1].set_ylabel('Số lượng', fontsize=12)
    axes[1].set_title('Phân Phối Thử Nghiệm Theo Phase', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_enrollment_duration_analysis(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Enrollment and duration relationship analysis
    
    Args:
        df: DataFrame with enrollment_count, trial_duration_days, and success columns
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Filter valid data
    df_filtered = df[
        (df['enrollment_count'].notna()) & 
        (df['trial_duration_days'].notna()) &
        (df['enrollment_count'] > 0) &
        (df['trial_duration_days'] > 0)
    ].copy()
    
    # Scatter plot
    for success_type in df_filtered['success'].unique():
        subset = df_filtered[df_filtered['success'] == success_type]
        color = '#2ecc71' if success_type == 'Thành công' else '#e74c3c'
        axes[0, 0].scatter(subset['enrollment_count'], subset['trial_duration_days'],
                          alpha=0.5, s=30, label=success_type, color=color)
    
    axes[0, 0].set_xlabel('Enrollment Count', fontsize=12)
    axes[0, 0].set_ylabel('Trial Duration (days)', fontsize=12)
    axes[0, 0].set_title('Enrollment vs Duration', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(alpha=0.3)
    
    # Box plots
    df_filtered.boxplot(column='enrollment_count', by='success', ax=axes[0, 1])
    axes[0, 1].set_ylabel('Enrollment Count (log)', fontsize=12)
    axes[0, 1].set_title('So Sánh Enrollment', fontsize=14, fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].get_figure().suptitle('')
    
    df_filtered.boxplot(column='trial_duration_days', by='success', ax=axes[1, 0])
    axes[1, 0].set_ylabel('Duration (days)', fontsize=12)
    axes[1, 0].set_title('So Sánh Duration', fontsize=14, fontweight='bold')
    axes[1, 0].get_figure().suptitle('')
    
    # Enrollment rate
    df_filtered['enrollment_rate'] = df_filtered['enrollment_count'] / df_filtered['trial_duration_days']
    for success_type in df_filtered['success'].unique():
        subset = df_filtered[df_filtered['success'] == success_type]
        color = '#2ecc71' if success_type == 'Thành công' else '#e74c3c'
        axes[1, 1].hist(subset['enrollment_rate'], alpha=0.6, bins=50,
                       color=color, label=success_type, density=True)
    
    axes[1, 1].set_xlabel('Enrollment Rate (patients/day)', fontsize=12)
    axes[1, 1].set_ylabel('Density', fontsize=12)
    axes[1, 1].set_title('Phân Phối Enrollment Rate', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, df_filtered['enrollment_rate'].quantile(0.95))
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_design_analysis(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Study design elements analysis
    
    Args:
        df: DataFrame with allocation, intervention_model, primary_purpose, and success
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Allocation analysis
    allocation_pivot = df.groupby(['allocation', 'success']).size().unstack(fill_value=0)
    allocation_pivot_pct = allocation_pivot.div(allocation_pivot.sum(axis=1), axis=0) * 100
    
    allocation_pivot_pct.plot(kind='barh', stacked=True, ax=axes[0, 0],
                              color=['#2ecc71', '#e74c3c'], width=0.7)
    axes[0, 0].set_xlabel('Tỷ lệ (%)', fontsize=12)
    axes[0, 0].set_title('Allocation vs Success', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Intervention model
    intervention_pivot = df.groupby(['intervention_model', 'success']).size().unstack(fill_value=0)
    intervention_pivot_pct = intervention_pivot.div(intervention_pivot.sum(axis=1), axis=0) * 100
    top_models = intervention_pivot.sum(axis=1).nlargest(8).index
    
    intervention_pivot_pct.loc[top_models].plot(kind='bar', ax=axes[0, 1],
                                                  color=['#2ecc71', '#e74c3c'], width=0.7)
    axes[0, 1].set_xlabel('Intervention Model', fontsize=12)
    axes[0, 1].set_ylabel('Tỷ lệ (%)', fontsize=12)
    axes[0, 1].set_title('Top Intervention Models', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Primary purpose
    purpose_pivot = df.groupby(['primary_purpose', 'success']).size().unstack(fill_value=0)
    purpose_pivot_pct = purpose_pivot.div(purpose_pivot.sum(axis=1), axis=0) * 100
    top_purposes = purpose_pivot.sum(axis=1).nlargest(10).index
    
    purpose_pivot_pct.loc[top_purposes].plot(kind='barh', ax=axes[1, 0],
                                              color=['#2ecc71', '#e74c3c'], width=0.7)
    axes[1, 0].set_xlabel('Tỷ lệ (%)', fontsize=12)
    axes[1, 0].set_title('Top Primary Purposes', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Heatmap - Allocation x Intervention
    failure_rates = df.groupby(['allocation', 'intervention_model'])['success'].apply(
        lambda x: (x == 'Thất bại').sum() / len(x) * 100 if len(x) > 0 else 0
    ).reset_index(name='failure_rate')
    
    pivot_heatmap = failure_rates.pivot(index='allocation', 
                                        columns='intervention_model',
                                        values='failure_rate')
    
    # Select top intervention models for heatmap
    top_cols = [col for col in top_models if col in pivot_heatmap.columns][:6]
    sns.heatmap(pivot_heatmap[top_cols], annot=True, fmt='.1f', cmap='RdYlGn_r',
                ax=axes[1, 1], cbar_kws={'label': 'Tỷ lệ thất bại (%)'})
    axes[1, 1].set_title('Allocation × Intervention Model', fontsize=14, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# PREPROCESSING VISUALIZATION FUNCTIONS
# =============================================================================

def plot_missing_rate_comparison(
    missing_before: pd.Series,
    missing_after: pd.Series,
    top_n: int = 20,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
):
    """
    Plot missing rate comparison before vs after cleaning (side-by-side bars).
    
    Args:
        missing_before: Series with missing rates before cleaning
        missing_after: Series with missing rates after cleaning  
        top_n: Number of top columns to show
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Combine and get top columns by before rate
    combined = pd.DataFrame({
        'before': missing_before,
        'after': missing_after
    }).fillna(0)
    
    # Sort by before and take top N
    top_cols = combined.nlargest(top_n, 'before')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(top_cols))
    width = 0.35
    
    bars1 = ax.barh(x - width/2, top_cols['before'] * 100, width, 
                    label='Before Cleaning', color='#e74c3c', alpha=0.7)
    bars2 = ax.barh(x + width/2, top_cols['after'] * 100, width,
                    label='After Cleaning', color='#2ecc71', alpha=0.7)
    
    ax.set_yticks(x)
    ax.set_yticklabels(top_cols.index, fontsize=10)
    ax.set_xlabel('Missing Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Missing Rate Before vs After Cleaning (Top {top_n} Columns)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 105)
    
    # Add threshold lines
    ax.axvline(x=30, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='30% threshold')
    ax.axvline(x=70, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='70% threshold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_enrollment_distribution(
    df: pd.DataFrame,
    enrollment_col: str = 'enrollment_count',
    log_col: str = 'log_enrollment',
    figsize: Tuple[int, int] = (16, 5),
    save_path: Optional[str] = None
):
    """
    Plot enrollment distribution with histogram and outlier info.
    
    Args:
        df: DataFrame with enrollment data
        enrollment_col: Name of enrollment column
        log_col: Name of log-transformed column
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    data = df[enrollment_col].dropna()
    
    # Plot 1: Raw distribution
    ax1 = axes[0]
    ax1.hist(data, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.0f}')
    ax1.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.0f}')
    ax1.set_xlabel('Enrollment Count', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Enrollment Distribution (Raw)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Log scale
    ax2 = axes[1]
    if log_col in df.columns:
        log_data = df[log_col].dropna()
    else:
        log_data = np.log1p(data)
    ax2.hist(log_data, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Log(1 + Enrollment)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Enrollment Distribution (Log Scale)', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Plot 3: Outlier analysis
    ax3 = axes[2]
    Q1, Q3 = data.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers_low = (data < lower).sum()
    outliers_high = (data > upper).sum()
    normal = len(data) - outliers_low - outliers_high
    
    categories = ['Normal Range', 'Low Outliers', 'High Outliers']
    counts = [normal, outliers_low, outliers_high]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax3.bar(categories, counts, color=colors, edgecolor='black')
    for bar, count in zip(bars, counts):
        pct = count / len(data) * 100
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(data)*0.01,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
    
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title(f'Outlier Distribution (IQR Method)\nBounds: [{lower:.0f}, {upper:.0f}]', 
                 fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Enrollment Statistics:")
    print(f"   P1: {data.quantile(0.01):.0f} | P5: {data.quantile(0.05):.0f} | P25: {Q1:.0f}")
    print(f"   Median: {data.median():.0f} | Mean: {data.mean():.0f}")
    print(f"   P75: {Q3:.0f} | P95: {data.quantile(0.95):.0f} | P99: {data.quantile(0.99):.0f}")


def plot_drop_reasons(
    drop_reasons: List[Dict],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot drop reasons as horizontal bar chart.
    
    Args:
        drop_reasons: List of dicts with 'Reason' and 'Count' keys
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if not drop_reasons:
        print("No drop reasons to plot.")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    df = pd.DataFrame(drop_reasons)
    df = df.sort_values('Count', ascending=True)
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(df)))
    
    bars = ax.barh(df['Reason'], df['Count'], color=colors, edgecolor='black')
    
    # Add count labels
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + df['Count'].max()*0.01, bar.get_y() + bar.get_height()/2,
               f"{row['Count']:,} ({row['Pct']})", va='center', fontsize=10)
    
    ax.set_xlabel('Number of Records Dropped', fontsize=12, fontweight='bold')
    ax.set_title('Records Dropped by Reason', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_label_distribution_by_phase(
    df: pd.DataFrame,
    phase_col: str = 'phases',
    label_col: str = 'label',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
):
    """
    Plot label distribution by phase (stacked bar chart).
    
    Args:
        df: DataFrame with phase and label columns
        phase_col: Name of phase column
        label_col: Name of label column
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Create crosstab
    ct = pd.crosstab(df[phase_col], df[label_col])
    ct.columns = ['Success (0)', 'Fail (1)']
    
    # Stacked count
    ct.plot(kind='bar', stacked=True, ax=axes[0], color=['#2ecc71', '#e74c3c'], 
            edgecolor='black', width=0.7)
    axes[0].set_xlabel('Phase', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Label Distribution by Phase (Count)', fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(title='Label')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Stacked percentage
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind='bar', stacked=True, ax=axes[1], color=['#2ecc71', '#e74c3c'],
               edgecolor='black', width=0.7)
    axes[1].set_xlabel('Phase', fontsize=12)
    axes[1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1].set_title('Label Distribution by Phase (Percentage)', fontsize=13, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Label')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add fail rate labels
    for i, (idx, row) in enumerate(ct_pct.iterrows()):
        fail_rate = row['Fail (1)']
        axes[1].annotate(f'{fail_rate:.1f}%', xy=(i, 50), ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_timeline_trials(
    df: pd.DataFrame,
    date_col: str = 'start_date',
    label_col: str = 'label',
    freq: str = 'M',
    cutoff_date: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
):
    """
    Plot timeline of trials by month/year with optional cutoff line.
    
    Args:
        df: DataFrame with date and label columns
        date_col: Name of date column
        label_col: Name of label column
        freq: Frequency for grouping ('M' for monthly, 'Y' for yearly)
        cutoff_date: Optional cutoff date for time-split visualization
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Ensure datetime
    df_plot = df.copy()
    df_plot[date_col] = pd.to_datetime(df_plot[date_col])
    df_plot = df_plot.dropna(subset=[date_col])
    
    # Group by period
    df_plot['period'] = df_plot[date_col].dt.to_period(freq)
    
    # Plot 1: Trial count over time
    ax1 = axes[0]
    counts = df_plot.groupby('period').size()
    counts.plot(kind='bar', ax=ax1, color='steelblue', alpha=0.7, width=0.8)
    ax1.set_xlabel('')
    ax1.set_ylabel('Number of Trials', fontsize=11)
    ax1.set_title(f'Number of Trials Over Time (by {freq})', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add cutoff line if provided
    if cutoff_date:
        cutoff = pd.Timestamp(cutoff_date)
        cutoff_period = cutoff.to_period(freq)
        if cutoff_period in counts.index:
            idx = list(counts.index).index(cutoff_period)
            ax1.axvline(x=idx, color='red', linestyle='--', linewidth=2, label=f'Cutoff: {cutoff_date}')
            ax1.legend()
    
    # Plot 2: Label ratio over time (rolling)
    ax2 = axes[1]
    label_ratio = df_plot.groupby('period')[label_col].mean() * 100
    label_ratio.plot(ax=ax2, color='#e74c3c', linewidth=2, marker='o', markersize=4)
    
    # Add rolling average
    if len(label_ratio) > 6:
        rolling_avg = label_ratio.rolling(window=6, min_periods=1).mean()
        ax2.plot(rolling_avg.index.astype(str), rolling_avg.values, 
                color='darkblue', linestyle='--', linewidth=2, label='6-period Rolling Avg')
        ax2.legend()
    
    ax2.set_xlabel(f'Time ({freq})', fontsize=11)
    ax2.set_ylabel('Fail Rate (%)', fontsize=11)
    ax2.set_title('Fail Rate Over Time (Label Drift Analysis)', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, min(100, label_ratio.max() * 1.3))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n💡 Insight: Check for label drift - if fail rate changes significantly over time,")
    print("   consider time-based split to avoid data leakage from temporal patterns.")


def plot_top_conditions(
    df: pd.DataFrame,
    conditions_col: str = 'conditions',
    top_n: int = 15,
    separator: str = '|',
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Plot top conditions by frequency (pipe-separated count).
    
    Args:
        df: DataFrame with conditions column
        conditions_col: Name of conditions column
        top_n: Number of top conditions to show
        separator: Separator for splitting conditions
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Extract and count conditions
    conditions_series = df[conditions_col].dropna()
    
    # Split by separator and flatten
    all_conditions = []
    for cond_str in conditions_series:
        if pd.notna(cond_str):
            conditions = [c.strip() for c in str(cond_str).split(separator) if c.strip()]
            all_conditions.extend(conditions)
    
    # Count frequency
    cond_counts = pd.Series(all_conditions).value_counts().head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cond_counts.plot(kind='barh', ax=ax, color='steelblue', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_ylabel('Condition', fontsize=12)
    ax.set_title(f'Top {top_n} Conditions by Frequency', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add count labels
    for bar in ax.patches:
        ax.text(bar.get_width() + cond_counts.max()*0.01, bar.get_y() + bar.get_height()/2,
               f'{int(bar.get_width()):,}', va='center', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Total unique conditions: {len(set(all_conditions)):,}")
    print(f"   Total condition mentions: {len(all_conditions):,}")


def plot_top_intervention_types(
    df: pd.DataFrame,
    interventions_col: str = 'interventions',
    top_n: int = 10,
    separator: str = '|',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot top intervention types by frequency.
    
    Args:
        df: DataFrame with interventions column
        interventions_col: Name of interventions column
        top_n: Number of top types to show
        separator: Separator for splitting
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Extract intervention types
    interventions_series = df[interventions_col].dropna()
    
    all_types = []
    for interv_str in interventions_series:
        if pd.notna(interv_str):
            # Extract type (typically format: "TYPE: Name")
            parts = str(interv_str).split(separator)
            for part in parts:
                if ':' in part:
                    itype = part.split(':')[0].strip().upper()
                else:
                    itype = part.strip().upper()
                if itype:
                    all_types.append(itype)
    
    type_counts = pd.Series(all_types).value_counts().head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
    type_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', width=0.7)
    
    ax.set_xlabel('Intervention Type', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Intervention Types', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    # Add count labels
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + type_counts.max()*0.01,
               f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_locations_distribution(
    df: pd.DataFrame,
    locations_col: str = 'locations_count',
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
):
    """
    Plot locations count distribution.
    
    Args:
        df: DataFrame with locations_count column
        locations_col: Name of locations count column
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    data = df[locations_col].dropna()
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(data, bins=50, color='teal', alpha=0.7, edgecolor='black')
    ax1.axvline(data.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {data.median():.0f}')
    ax1.set_xlabel('Number of Locations', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Locations Count Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Log scale histogram
    ax2 = axes[1]
    ax2.hist(np.log1p(data), bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Log(1 + Locations)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Locations (Log Scale)', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Category breakdown
    ax3 = axes[2]
    categories = pd.cut(data, bins=[0, 1, 5, 20, 100, np.inf], 
                       labels=['Single (1)', '2-5', '6-20', '21-100', '>100'])
    cat_counts = categories.value_counts().sort_index()
    
    cat_counts.plot(kind='bar', ax=ax3, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(cat_counts))),
                   edgecolor='black', width=0.7)
    ax3.set_xlabel('Location Category', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Trials by Location Count Category', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add percentages
    for bar in ax3.patches:
        pct = bar.get_height() / len(data) * 100
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(data)*0.01,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_missingness_heatmap(
    df: pd.DataFrame,
    sample_size: int = 300,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
):
    """
    Plot missingness pattern heatmap for sample rows.
    
    Args:
        df: DataFrame to analyze
        sample_size: Number of rows to sample
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Select columns with some missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    
    if not missing_cols:
        print("No missing values found in the dataset!")
        return
    
    # Sample rows
    sample_df = df[missing_cols].sample(min(sample_size, len(df)), random_state=42)
    
    # Create missing indicator matrix
    missing_matrix = sample_df.isnull().astype(int)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(missing_matrix.T, cmap='YlOrRd', cbar_kws={'label': 'Missing'},
               xticklabels=False, yticklabels=True, ax=ax)
    
    ax.set_xlabel(f'Sample Rows (n={len(sample_df)})', fontsize=11)
    ax.set_ylabel('Columns', fontsize=11)
    ax.set_title('Missingness Pattern Heatmap (Yellow=Missing)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Showing {len(missing_cols)} columns with missing values across {len(sample_df)} sample rows")


def plot_split_sanity_table(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    label_col: str = 'label',
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
):
    """
    Display split sanity check table and visualization.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        val_df: Optional validation DataFrame
        label_col: Name of label column
        figsize: Figure size
        save_path: Optional path to save figure
    """
    splits = [('Train', train_df), ('Test', test_df)]
    if val_df is not None and len(val_df) > 0:
        splits.insert(1, ('Validation', val_df))
    
    # Build summary table
    summary = []
    for name, df in splits:
        n = len(df)
        n_fail = (df[label_col] == 1).sum() if label_col in df.columns else 0
        n_success = (df[label_col] == 0).sum() if label_col in df.columns else 0
        fail_rate = n_fail / n * 100 if n > 0 else 0
        
        summary.append({
            'Split': name,
            'N_Samples': n,
            'Fail (1)': n_fail,
            'Success (0)': n_success,
            'Fail Rate': f'{fail_rate:.2f}%'
        })
    
    summary_df = pd.DataFrame(summary)
    
    # Print table
    print("\n" + "=" * 60)
    print("📊 SPLIT SANITY TABLE")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    
    # Check for duplicate IDs
    if 'nct_id' in train_df.columns and 'nct_id' in test_df.columns:
        train_ids = set(train_df['nct_id'])
        test_ids = set(test_df['nct_id'])
        overlap = train_ids & test_ids
        
        print("\n🔍 NCT_ID Overlap Check:")
        print(f"   Train IDs: {len(train_ids):,}")
        print(f"   Test IDs: {len(test_ids):,}")
        print(f"   Overlap: {len(overlap)}")
        
        if len(overlap) == 0:
            print("   ✅ PASS: No overlap between train and test!")
        else:
            print(f"   ❌ FAIL: {len(overlap)} overlapping IDs!")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Size comparison
    ax1 = axes[0]
    names = [s['Split'] for s in summary]
    sizes = [s['N_Samples'] for s in summary]
    colors = ['#3498db', '#f39c12', '#e74c3c'][:len(names)]
    
    bars = ax1.bar(names, sizes, color=colors, edgecolor='black')
    for bar, size in zip(bars, sizes):
        total = sum(sizes)
        pct = size / total * 100
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.02,
                f'{size:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
    
    ax1.set_ylabel('Number of Samples', fontsize=11)
    ax1.set_title('Split Sizes', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Fail rate comparison
    ax2 = axes[1]
    fail_rates = [float(s['Fail Rate'].replace('%', '')) for s in summary]
    
    bars = ax2.bar(names, fail_rates, color=['#e74c3c']*len(names), alpha=0.7, edgecolor='black')
    for bar, rate in zip(bars, fail_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.2f}%', ha='center', va='bottom', fontsize=10)
    
    ax2.set_ylabel('Fail Rate (%)', fontsize=11)
    ax2.set_title('Fail Rate by Split', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, max(fail_rates) * 1.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return summary_df
