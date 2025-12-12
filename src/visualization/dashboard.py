from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings

warnings.filterwarnings("ignore")


class InteractiveDashboard:
    """
    Create interactive visualizations and dashboards for data analysis.
    """

    def __init__(self, theme: str = "plotly_white") -> None:
        """
        Initialize InteractiveDashboard.

        Args:
            theme: Plotly theme, e.g. 'plotly', 'plotly_white', 'plotly_dark',
                   'ggplot2', 'seaborn', ...
        """
        self.theme = theme
        self.default_height: int = 600
        self.default_width: int = 1000

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _apply_theme(self, fig: go.Figure) -> go.Figure:
        """Apply theme and common layout settings."""
        fig.update_layout(
            template=self.theme,
            font=dict(size=12),
            title_font=dict(size=16, family="Arial Black"),
            hovermode="closest",
            showlegend=True,
        )
        return fig

    # ==================================================================
    # DISTRIBUTION VISUALIZATIONS
    # ==================================================================

    def plot_interactive_histogram(
        self,
        df: pd.DataFrame,
        column: str,
        color_by: Optional[str] = None,
        bins: int = 30,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive histogram with optional color grouping.

        Args:
            df: Input DataFrame.
            column: Column to plot on x-axis.
            color_by: Column to group colors by (optional).
            bins: Number of bins.
            title: Plot title (optional).

        Returns:
            Plotly Figure object.
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        if color_by is not None and color_by not in df.columns:
            raise ValueError(f"color_by column '{color_by}' not found in DataFrame.")

        title = title or f"Distribution of {column}"

        if color_by:
            fig = px.histogram(
                df,
                x=column,
                color=color_by,
                nbins=bins,
                title=title,
                marginal="box",  # Add box plot on top
                hover_data=df.columns,
            )
        else:
            fig = px.histogram(
                df,
                x=column,
                nbins=bins,
                title=title,
                marginal="box",
                hover_data=df.columns,
            )

        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Count",
            height=self.default_height,
            width=self.default_width,
        )

        return self._apply_theme(fig)

    def plot_interactive_box(
        self,
        df: pd.DataFrame,
        y_col: str,
        x_col: Optional[str] = None,
        color_by: Optional[str] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive box plot.

        Args:
            df: Input DataFrame.
            y_col: Column for y-axis (numeric).
            x_col: Column for x-axis (categorical, optional).
            color_by: Column for color grouping (optional).
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        if y_col not in df.columns:
            raise ValueError(f"y_col '{y_col}' not found in DataFrame.")
        if x_col is not None and x_col not in df.columns:
            raise ValueError(f"x_col '{x_col}' not found in DataFrame.")
        if color_by is not None and color_by not in df.columns:
            raise ValueError(f"color_by '{color_by}' not found in DataFrame.")

        title = title or f"Distribution of {y_col}" + (f" by {x_col}" if x_col else "")

        fig = px.box(
            df,
            x=x_col,
            y=y_col,
            color=color_by,
            title=title,
            hover_data=df.columns,
            points="outliers",  # Show outlier points
        )

        fig.update_layout(height=self.default_height, width=self.default_width)

        return self._apply_theme(fig)

    def plot_interactive_violin(
        self,
        df: pd.DataFrame,
        y_col: str,
        x_col: Optional[str] = None,
        color_by: Optional[str] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive violin plot.

        Args:
            df: Input DataFrame.
            y_col: Column for y-axis (numeric).
            x_col: Column for x-axis (categorical, optional).
            color_by: Column for color grouping (optional).
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        if y_col not in df.columns:
            raise ValueError(f"y_col '{y_col}' not found in DataFrame.")
        if x_col is not None and x_col not in df.columns:
            raise ValueError(f"x_col '{x_col}' not found in DataFrame.")
        if color_by is not None and color_by not in df.columns:
            raise ValueError(f"color_by '{color_by}' not found in DataFrame.")

        title = title or f"Distribution of {y_col}" + (f" by {x_col}" if x_col else "")

        fig = px.violin(
            df,
            x=x_col,
            y=y_col,
            color=color_by,
            title=title,
            box=True,  # Add box plot inside violin
            hover_data=df.columns,
        )

        fig.update_layout(height=self.default_height, width=self.default_width)

        return self._apply_theme(fig)

    # ==================================================================
    # COMPARISON VISUALIZATIONS
    # ==================================================================

    def plot_interactive_bar(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: Optional[str] = None,
        color_by: Optional[str] = None,
        orientation: str = "v",
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive bar chart.

        Args:
            df: Input DataFrame.
            x_col: Column for x-axis (or y-axis if orientation='h').
            y_col: Column for y-axis (if None, uses count of x_col).
            color_by: Column for color grouping.
            orientation: 'v' for vertical, 'h' for horizontal.
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        if x_col not in df.columns:
            raise ValueError(f"x_col '{x_col}' not found in DataFrame.")
        if y_col is not None and y_col not in df.columns:
            raise ValueError(f"y_col '{y_col}' not found in DataFrame.")
        if color_by is not None and color_by not in df.columns:
            raise ValueError(f"color_by '{color_by}' not found in DataFrame.")
        if orientation not in {"v", "h"}:
            raise ValueError("orientation must be 'v' or 'h'.")

        if y_col is None:
            # Count plot
            value_counts = df[x_col].value_counts().reset_index()
            value_counts.columns = [x_col, "count"]
            y_col_internal = "count"
            plot_data = value_counts
        else:
            plot_data = df
            y_col_internal = y_col

        title = title or f"{y_col_internal} by {x_col}"

        fig = px.bar(
            plot_data,
            x=x_col if orientation == "v" else y_col_internal,
            y=y_col_internal if orientation == "v" else x_col,
            color=color_by,
            title=title,
            orientation=orientation,
            hover_data=plot_data.columns,
        )

        fig.update_layout(height=self.default_height, width=self.default_width)

        return self._apply_theme(fig)

    def plot_grouped_bar(
        self,
        df: pd.DataFrame,
        categorical_col: str,
        value_col: str,
        group_by: str,
        aggregation: str = "mean",
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create grouped bar chart with aggregation.

        Args:
            df: Input DataFrame.
            categorical_col: Column for x-axis categories.
            value_col: Column to aggregate.
            group_by: Column to group bars by.
            aggregation: Aggregation method ('mean', 'sum', 'count', 'median').
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        for col in (categorical_col, value_col, group_by):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        aggregation = aggregation.lower()
        allowed_aggs = {"mean", "sum", "count", "median"}
        if aggregation not in allowed_aggs:
            raise ValueError(f"aggregation must be one of {allowed_aggs}, got '{aggregation}'.")

        # Perform aggregation
        if aggregation == "mean":
            agg_df = df.groupby([categorical_col, group_by])[value_col].mean().reset_index()
        elif aggregation == "sum":
            agg_df = df.groupby([categorical_col, group_by])[value_col].sum().reset_index()
        elif aggregation == "count":
            agg_df = df.groupby([categorical_col, group_by])[value_col].count().reset_index()
        else:  # median
            agg_df = df.groupby([categorical_col, group_by])[value_col].median().reset_index()

        title = title or f"{aggregation.capitalize()} {value_col} by {categorical_col} and {group_by}"

        fig = px.bar(
            agg_df,
            x=categorical_col,
            y=value_col,
            color=group_by,
            title=title,
            barmode="group",
            text_auto=".2f",
        )

        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            yaxis_title=f"{aggregation.capitalize()} {value_col}",
        )

        return self._apply_theme(fig)

    # ==================================================================
    # RELATIONSHIP VISUALIZATIONS
    # ==================================================================

    def plot_interactive_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_by: Optional[str] = None,
        size_by: Optional[str] = None,
        add_trendline: bool = False,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive scatter plot.

        Args:
            df: Input DataFrame.
            x_col: Column for x-axis.
            y_col: Column for y-axis.
            color_by: Column for color grouping (optional).
            size_by: Column for point size (optional).
            add_trendline: Whether to add OLS trendline.
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        for col in (x_col, y_col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        if color_by is not None and color_by not in df.columns:
            raise ValueError(f"color_by '{color_by}' not found in DataFrame.")
        if size_by is not None and size_by not in df.columns:
            raise ValueError(f"size_by '{size_by}' not found in DataFrame.")

        title = title or f"{y_col} vs {x_col}"

        trendline = "ols" if add_trendline else None

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_by,
            size=size_by,
            title=title,
            trendline=trendline,
            hover_data=df.columns,
        )

        # Add correlation annotation (safe)
        corr_text = "Correlation: N/A"
        try:
            corr_df = df[[x_col, y_col]].dropna()
            if len(corr_df) >= 2 and np.issubdtype(corr_df[x_col].dtype, np.number) and np.issubdtype(
                corr_df[y_col].dtype, np.number
            ):
                corr_value = corr_df[x_col].corr(corr_df[y_col])
                if pd.notna(corr_value):
                    corr_text = f"Correlation: {corr_value:.3f}"
        except Exception:
            pass

        fig.add_annotation(
            text=corr_text,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            bgcolor="lightgray",
            opacity=0.8,
        )

        fig.update_layout(height=self.default_height, width=self.default_width)

        return self._apply_theme(fig)

    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        columns: Optional[Sequence[str]] = None,
        method: str = "pearson",
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive correlation heatmap.

        Args:
            df: Input DataFrame.
            columns: Columns to include (None = all numeric).
            method: Correlation method ('pearson', 'spearman').
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        if columns is not None:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"Columns {missing} not found in DataFrame.")
            corr_data = df[list(columns)]
        else:
            corr_data = df.select_dtypes(include=[np.number])

        if corr_data.shape[1] == 0:
            raise ValueError("No numeric columns available for correlation matrix.")

        corr_matrix = corr_data.corr(method=method)

        title = title or f"Correlation Matrix ({method.capitalize()})"

        fig = px.imshow(
            corr_matrix,
            title=title,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            text_auto=".2f",
            zmin=-1,
            zmax=1,
        )

        size = max(600, corr_matrix.shape[0] * 30)
        fig.update_layout(
            height=size,
            width=size,
        )

        return self._apply_theme(fig)

    # ==================================================================
    # TIME SERIES VISUALIZATIONS
    # ==================================================================

    def plot_time_series(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        group_by: Optional[str] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive time series plot.

        Args:
            df: Input DataFrame.
            date_col: Column with dates.
            value_col: Column with values.
            group_by: Column to group lines by (multiple series).
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        for col in (date_col, value_col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        if group_by is not None and group_by not in df.columns:
            raise ValueError(f"group_by '{group_by}' not found in DataFrame.")

        title = title or f"{value_col} Over Time"

        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")

        fig = px.line(
            df_copy,
            x=date_col,
            y=value_col,
            color=group_by,
            title=title,
            markers=True,
            hover_data=df_copy.columns,
        )

        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            xaxis_title=date_col,
            yaxis_title=value_col,
        )

        fig.update_xaxes(rangeslider_visible=True)

        return self._apply_theme(fig)

    def plot_time_aggregation(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        aggregation: str = "mean",
        freq: str = "M",
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create aggregated time series with interactive controls.

        Args:
            df: Input DataFrame.
            date_col: Column with dates.
            value_col: Column with values.
            aggregation: Aggregation method ('mean', 'sum', 'count', 'median').
            freq: Frequency string ('D'=daily, 'W'=weekly, 'M'=monthly, 'Y'=yearly, ...).
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        for col in (date_col, value_col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        aggregation = aggregation.lower()
        allowed_aggs = {"mean", "sum", "count", "median"}
        if aggregation not in allowed_aggs:
            raise ValueError(f"aggregation must be one of {allowed_aggs}, got '{aggregation}'.")

        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
        df_copy = df_copy.set_index(date_col)

        if aggregation == "mean":
            agg_series = df_copy[value_col].resample(freq).mean()
        elif aggregation == "sum":
            agg_series = df_copy[value_col].resample(freq).sum()
        elif aggregation == "count":
            agg_series = df_copy[value_col].resample(freq).count()
        else:  # median
            agg_series = df_copy[value_col].resample(freq).median()

        agg_df = agg_series.reset_index()

        title = title or f"{aggregation.capitalize()} {value_col} by {freq}"

        fig = go.Figure()

        # Actual aggregated values
        fig.add_trace(
            go.Scatter(
                x=agg_df[date_col],
                y=agg_df[value_col],
                mode="lines+markers",
                name="Actual",
                line=dict(width=2),
                marker=dict(size=6),
            )
        )

        # Simple linear trend line
        if len(agg_df) >= 2:
            x_numeric = np.arange(len(agg_df))
            z = np.polyfit(x_numeric, agg_df[value_col].values, 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=agg_df[date_col],
                    y=p(x_numeric),
                    mode="lines",
                    name="Trend",
                    line=dict(width=2, dash="dash"),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title=date_col,
            yaxis_title=f"{aggregation.capitalize()} {value_col}",
            height=self.default_height,
            width=self.default_width,
            hovermode="x unified",
        )

        fig.update_xaxes(rangeslider_visible=True)

        return self._apply_theme(fig)

    # ==================================================================
    # COMPOSITION VISUALIZATIONS
    # ==================================================================

    def plot_pie_chart(
        self,
        df: pd.DataFrame,
        column: str,
        top_n: Optional[int] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive (donut) pie chart.

        Args:
            df: Input DataFrame.
            column: Column to visualize (categorical).
            top_n: Show only top N categories (rest aggregated to 'Other').
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        value_counts = df[column].value_counts()

        if top_n is not None and top_n > 0 and len(value_counts) > top_n:
            top = value_counts.head(top_n)
            others = value_counts.iloc[top_n:].sum()
            value_counts = top.append(pd.Series({"Other": others}))

        title = title or f"Distribution of {column}"

        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=title,
            hole=0.3,  # Make it a donut chart
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}",
        )

        fig.update_layout(height=self.default_height, width=self.default_width)

        return self._apply_theme(fig)

    def plot_sunburst(
        self,
        df: pd.DataFrame,
        path_columns: List[str],
        value_col: Optional[str] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create hierarchical sunburst chart.

        Args:
            df: Input DataFrame.
            path_columns: Columns defining hierarchy (from outer to inner).
            value_col: Column for values (if None, uses count).
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        for col in path_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        title = title or f"Hierarchical View: {' → '.join(path_columns)}"

        if value_col is not None:
            if value_col not in df.columns:
                raise ValueError(f"value_col '{value_col}' not found in DataFrame.")
            fig = px.sunburst(df, path=path_columns, values=value_col, title=title)
        else:
            count_df = df.groupby(path_columns).size().reset_index(name="count")
            fig = px.sunburst(count_df, path=path_columns, values="count", title=title)

        fig.update_layout(height=self.default_height, width=self.default_width)

        return self._apply_theme(fig)

    def plot_treemap(
        self,
        df: pd.DataFrame,
        path_columns: List[str],
        value_col: Optional[str] = None,
        color_col: Optional[str] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create hierarchical treemap.

        Args:
            df: Input DataFrame.
            path_columns: Columns defining hierarchy.
            value_col: Column for values (if None, uses count).
            color_col: Column for coloring.
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        for col in path_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        if color_col is not None and color_col not in df.columns:
            raise ValueError(f"color_col '{color_col}' not found in DataFrame.")

        title = title or f"Treemap: {' → '.join(path_columns)}"

        if value_col is None:
            count_df = df.groupby(path_columns).size().reset_index(name="count")
            value_col_internal = "count"
            plot_df = count_df
        else:
            if value_col not in df.columns:
                raise ValueError(f"value_col '{value_col}' not found in DataFrame.")
            value_col_internal = value_col
            plot_df = df

        fig = px.treemap(
            plot_df,
            path=path_columns,
            values=value_col_internal,
            color=color_col,
            title=title,
        )

        fig.update_layout(height=self.default_height, width=self.default_width)

        return self._apply_theme(fig)

    # ==================================================================
    # ADVANCED VISUALIZATIONS
    # ==================================================================

    def plot_parallel_coordinates(
        self,
        df: pd.DataFrame,
        dimensions: List[str],
        color_by: str,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create parallel coordinates plot for multivariate analysis.

        Args:
            df: Input DataFrame.
            dimensions: List of numeric columns to include.
            color_by: Column to color by.
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        for col in dimensions + [color_by]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        title = title or "Parallel Coordinates Plot"

        fig = px.parallel_coordinates(df, dimensions=dimensions, color=color_by, title=title)

        fig.update_layout(height=self.default_height, width=self.default_width)

        return self._apply_theme(fig)

    def plot_parallel_categories(
        self,
        df: pd.DataFrame,
        dimensions: List[str],
        color_by: Optional[str] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create parallel categories (Sankey-style) plot.

        Args:
            df: Input DataFrame.
            dimensions: List of categorical columns.
            color_by: Column to color by (optional).
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        for col in dimensions:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        if color_by is not None and color_by not in df.columns:
            raise ValueError(f"color_by '{color_by}' not found in DataFrame.")

        title = title or f"Flow Analysis: {' → '.join(dimensions)}"

        fig = px.parallel_categories(df, dimensions=dimensions, color=color_by, title=title)

        fig.update_layout(height=self.default_height, width=self.default_width)

        return self._apply_theme(fig)

    def plot_3d_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        color_by: Optional[str] = None,
        size_by: Optional[str] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create 3D scatter plot.

        Args:
            df: Input DataFrame.
            x_col: Column for x-axis.
            y_col: Column for y-axis.
            z_col: Column for z-axis.
            color_by: Column for color grouping (optional).
            size_by: Column for point size (optional).
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        for col in (x_col, y_col, z_col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        if color_by is not None and color_by not in df.columns:
            raise ValueError(f"color_by '{color_by}' not found in DataFrame.")
        if size_by is not None and size_by not in df.columns:
            raise ValueError(f"size_by '{size_by}' not found in DataFrame.")

        title = title or f"3D Scatter: {x_col}, {y_col}, {z_col}"

        fig = px.scatter_3d(
            df,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_by,
            size=size_by,
            title=title,
            hover_data=df.columns,
        )

        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col),
        )

        return self._apply_theme(fig)

    # ==================================================================
    # DASHBOARD LAYOUTS
    # ==================================================================

    def create_overview_dashboard(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str],
        target_col: str,
    ) -> go.Figure:
        """
        Create a basic overview dashboard with multiple subplots.

        Layout:
        - Row 1, Col 1: Target distribution (pie)
        - Row 1, Col 2: Top categories of first categorical feature
        - Row 2, Col 1: Boxplots for first numerical features
        - Row 2, Col 2: Correlation with numeric target (if applicable)

        Args:
            df: Input DataFrame.
            numerical_cols: Numeric feature columns.
            categorical_cols: Categorical feature columns.
            target_col: Target variable column.

        Returns:
            Plotly Figure with subplots.
        """
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in DataFrame.")

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Target Distribution",
                "Top Categories",
                "Numerical Feature Distribution",
                "Correlation with Target",
            ),
            specs=[[{"type": "pie"}, {"type": "bar"}], [{"type": "box"}, {"type": "bar"}]],
        )

        # 1. Target distribution
        target_counts = df[target_col].value_counts()
        fig.add_trace(
            go.Pie(labels=target_counts.index, values=target_counts.values, hole=0.3, name="Target"),
            row=1,
            col=1,
        )

        # 2. Top categories for the first categorical column
        if categorical_cols:
            cat_col = categorical_cols[0]
            if cat_col in df.columns:
                cat_counts = df[cat_col].value_counts().head(10)
                fig.add_trace(
                    go.Bar(x=cat_counts.index, y=cat_counts.values, name=cat_col),
                    row=1,
                    col=2,
                )

        # 3. Numerical distributions (boxplots)
        if numerical_cols:
            for num_col in numerical_cols[:3]:
                if num_col in df.columns:
                    fig.add_trace(go.Box(y=df[num_col], name=num_col), row=2, col=1)

        # 4. Correlation with numeric target
        if numerical_cols and pd.api.types.is_numeric_dtype(df[target_col]):
            corr_series = df[numerical_cols].corrwith(df[target_col]).dropna()
            corr_series = corr_series.reindex(
                corr_series.abs().sort_values(ascending=False).index
            ).head(10)

            if not corr_series.empty:
                fig.add_trace(
                    go.Bar(
                        x=corr_series.values,
                        y=corr_series.index,
                        orientation="h",
                        marker_color=["red" if x < 0 else "green" for x in corr_series.values],
                        name="Correlation",
                    ),
                    row=2,
                    col=2,
                )

        fig.update_layout(
            title_text="Data Overview Dashboard",
            title_font_size=20,
            showlegend=True,
            height=800,
            width=1200,
        )

        return self._apply_theme(fig)

    # ==================================================================
    # SAVE FIGURE
    # ==================================================================

    def save_figure(self, fig: go.Figure, filename: str, format: str = "html") -> None:
        """
        Save Plotly figure to file.

        Args:
            fig: Plotly Figure object.
            filename: Output filename (with extension).
            format: Output format ('html', 'png', 'jpg', 'svg', 'pdf').

        Note:
            For non-HTML formats, Plotly may require the 'kaleido' package installed.
        """
        format = format.lower()
        if format == "html":
            fig.write_html(filename)
        else:
            fig.write_image(filename)
        print(f"Figure saved to: {filename}")


# =====================================================================
# CONVENIENCE FUNCTION
# =====================================================================

def create_quick_dashboard(df: pd.DataFrame, target_col: str) -> go.Figure:
    """
    Quickly create an overview dashboard for a given dataset.

    Args:
        df: Input DataFrame.
        target_col: Target variable column name.

    Returns:
        Plotly Figure object.
    """
    dashboard = InteractiveDashboard()

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    return dashboard.create_overview_dashboard(df, numerical_cols, categorical_cols, target_col)
