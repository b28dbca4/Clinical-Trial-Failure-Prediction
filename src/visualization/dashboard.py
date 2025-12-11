"""
Interactive Dashboard Module for Clinical Trial Failure Prediction

This module provides interactive visualizations using Plotly for:
- Dynamic data exploration
- Interactive filtering and drill-down analysis
- Presenting insights in an engaging way
- Answering meaningful questions with interactive elements

Designed for presentations and interactive exploration during the final seminar.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Union
import warnings

warnings.filterwarnings('ignore')


class InteractiveDashboard:
    """Create interactive visualizations and dashboards for clinical trial analysis"""
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize InteractiveDashboard
        
        Args:
            theme: Plotly theme ('plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn')
        """
        self.theme = theme
        self.default_height = 600
        self.default_width = 1000
    
    def _apply_theme(self, fig: go.Figure) -> go.Figure:
        """Apply theme and common layout settings"""
        fig.update_layout(
            template=self.theme,
            font=dict(size=12),
            title_font=dict(size=16, family='Arial Black'),
            hovermode='closest',
            showlegend=True
        )
        return fig
    
    # ==================== DISTRIBUTION VISUALIZATIONS ====================
    
    def plot_interactive_histogram(
        self,
        data: pd.DataFrame,
        column: str,
        color_by: Optional[str] = None,
        bins: int = 30,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive histogram with optional color grouping
        
        Args:
            data: DataFrame
            column: Column to plot
            color_by: Column to group colors by
            bins: Number of bins
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        title = title or f'Distribution of {column}'
        
        if color_by:
            fig = px.histogram(
                data, 
                x=column, 
                color=color_by,
                nbins=bins,
                title=title,
                marginal='box',  # Add box plot on top
                hover_data=data.columns
            )
        else:
            fig = px.histogram(
                data, 
                x=column,
                nbins=bins,
                title=title,
                marginal='box'
            )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title='Count',
            height=self.default_height
        )
        
        return self._apply_theme(fig)
    
    def plot_interactive_box(
        self,
        data: pd.DataFrame,
        y_col: str,
        x_col: Optional[str] = None,
        color_by: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive box plot
        
        Args:
            data: DataFrame
            y_col: Column for y-axis (numerical)
            x_col: Column for x-axis (categorical)
            color_by: Column for color grouping
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        title = title or f'Distribution of {y_col}' + (f' by {x_col}' if x_col else '')
        
        fig = px.box(
            data,
            x=x_col,
            y=y_col,
            color=color_by,
            title=title,
            hover_data=data.columns,
            points='outliers'  # Show outlier points
        )
        
        fig.update_layout(height=self.default_height)
        
        return self._apply_theme(fig)
    
    def plot_interactive_violin(
        self,
        data: pd.DataFrame,
        y_col: str,
        x_col: Optional[str] = None,
        color_by: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive violin plot
        
        Args:
            data: DataFrame
            y_col: Column for y-axis (numerical)
            x_col: Column for x-axis (categorical)
            color_by: Column for color grouping
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        title = title or f'Distribution of {y_col}' + (f' by {x_col}' if x_col else '')
        
        fig = px.violin(
            data,
            x=x_col,
            y=y_col,
            color=color_by,
            title=title,
            box=True,  # Add box plot inside violin
            hover_data=data.columns
        )
        
        fig.update_layout(height=self.default_height)
        
        return self._apply_theme(fig)
    
    # ==================== COMPARISON VISUALIZATIONS ====================
    
    def plot_interactive_bar(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: Optional[str] = None,
        color_by: Optional[str] = None,
        orientation: str = 'v',
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive bar chart
        
        Args:
            data: DataFrame
            x_col: Column for x-axis
            y_col: Column for y-axis (if None, uses count)
            color_by: Column for color grouping
            orientation: 'v' for vertical, 'h' for horizontal
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if y_col is None:
            # Count plot
            value_counts = data[x_col].value_counts().reset_index()
            value_counts.columns = [x_col, 'count']
            y_col = 'count'
            plot_data = value_counts
        else:
            plot_data = data
        
        title = title or f'{y_col} by {x_col}'
        
        fig = px.bar(
            plot_data,
            x=x_col if orientation == 'v' else y_col,
            y=y_col if orientation == 'v' else x_col,
            color=color_by,
            title=title,
            orientation=orientation,
            hover_data=plot_data.columns
        )
        
        fig.update_layout(height=self.default_height)
        
        return self._apply_theme(fig)
    
    def plot_grouped_bar(
        self,
        data: pd.DataFrame,
        categorical_col: str,
        value_col: str,
        group_by: str,
        aggregation: str = 'mean',
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create grouped bar chart with aggregation
        
        Args:
            data: DataFrame
            categorical_col: Column for x-axis categories
            value_col: Column to aggregate
            group_by: Column to group bars by
            aggregation: Aggregation method ('mean', 'sum', 'count', 'median')
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        # Perform aggregation
        if aggregation == 'mean':
            agg_data = data.groupby([categorical_col, group_by])[value_col].mean().reset_index()
        elif aggregation == 'sum':
            agg_data = data.groupby([categorical_col, group_by])[value_col].sum().reset_index()
        elif aggregation == 'count':
            agg_data = data.groupby([categorical_col, group_by])[value_col].count().reset_index()
        elif aggregation == 'median':
            agg_data = data.groupby([categorical_col, group_by])[value_col].median().reset_index()
        
        title = title or f'{aggregation.capitalize()} {value_col} by {categorical_col} and {group_by}'
        
        fig = px.bar(
            agg_data,
            x=categorical_col,
            y=value_col,
            color=group_by,
            title=title,
            barmode='group',
            text_auto='.2f'
        )
        
        fig.update_layout(
            height=self.default_height,
            yaxis_title=f'{aggregation.capitalize()} {value_col}'
        )
        
        return self._apply_theme(fig)
    
    # ==================== RELATIONSHIP VISUALIZATIONS ====================
    
    def plot_interactive_scatter(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_by: Optional[str] = None,
        size_by: Optional[str] = None,
        add_trendline: bool = False,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive scatter plot
        
        Args:
            data: DataFrame
            x_col: Column for x-axis
            y_col: Column for y-axis
            color_by: Column for color grouping
            size_by: Column for point size
            add_trendline: Whether to add trendline
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        title = title or f'{y_col} vs {x_col}'
        
        trendline = 'ols' if add_trendline else None
        
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color_by,
            size=size_by,
            title=title,
            trendline=trendline,
            hover_data=data.columns
        )
        
        # Add correlation annotation
        corr = data[[x_col, y_col]].corr().iloc[0, 1]
        fig.add_annotation(
            text=f'Correlation: {corr:.3f}',
            xref='paper', yref='paper',
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor='lightgray',
            opacity=0.8
        )
        
        fig.update_layout(height=self.default_height)
        
        return self._apply_theme(fig)
    
    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'pearson',
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive correlation heatmap
        
        Args:
            data: DataFrame
            columns: Columns to include (None = all numeric)
            method: Correlation method ('pearson', 'spearman')
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if columns:
            corr_data = data[columns]
        else:
            corr_data = data.select_dtypes(include=[np.number])
        
        corr_matrix = corr_data.corr(method=method)
        
        title = title or f'Correlation Matrix ({method.capitalize()})'
        
        fig = px.imshow(
            corr_matrix,
            title=title,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            text_auto='.2f',
            zmin=-1,
            zmax=1
        )
        
        fig.update_layout(
            height=max(600, len(corr_matrix) * 30),
            width=max(700, len(corr_matrix) * 30)
        )
        
        return self._apply_theme(fig)
    
    # ==================== TIME SERIES VISUALIZATIONS ====================
    
    def plot_time_series(
        self,
        data: pd.DataFrame,
        date_col: str,
        value_col: str,
        group_by: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive time series plot
        
        Args:
            data: DataFrame
            date_col: Column with dates
            value_col: Column with values
            group_by: Column to group by (creates multiple lines)
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        title = title or f'{value_col} Over Time'
        
        fig = px.line(
            data,
            x=date_col,
            y=value_col,
            color=group_by,
            title=title,
            markers=True,
            hover_data=data.columns
        )
        
        fig.update_layout(
            height=self.default_height,
            xaxis_title=date_col,
            yaxis_title=value_col
        )
        
        # Add range slider
        fig.update_xaxes(rangeslider_visible=True)
        
        return self._apply_theme(fig)
    
    def plot_time_aggregation(
        self,
        data: pd.DataFrame,
        date_col: str,
        value_col: str,
        aggregation: str = 'mean',
        freq: str = 'M',
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create aggregated time series with interactive controls
        
        Args:
            data: DataFrame
            date_col: Column with dates
            value_col: Column with values
            aggregation: Aggregation method ('mean', 'sum', 'count', 'median')
            freq: Frequency ('D'=daily, 'W'=weekly, 'M'=monthly, 'Y'=yearly)
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        
        # Aggregate
        if aggregation == 'mean':
            agg_data = df[value_col].resample(freq).mean()
        elif aggregation == 'sum':
            agg_data = df[value_col].resample(freq).sum()
        elif aggregation == 'count':
            agg_data = df[value_col].resample(freq).count()
        elif aggregation == 'median':
            agg_data = df[value_col].resample(freq).median()
        
        agg_df = agg_data.reset_index()
        
        title = title or f'{aggregation.capitalize()} {value_col} by {freq}'
        
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=agg_df[date_col],
            y=agg_df[value_col],
            mode='lines+markers',
            name='Actual',
            line=dict(color='steelblue', width=2),
            marker=dict(size=6)
        ))
        
        # Add trend line
        x_numeric = np.arange(len(agg_df))
        z = np.polyfit(x_numeric, agg_df[value_col].values, 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=agg_df[date_col],
            y=p(x_numeric),
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=date_col,
            yaxis_title=f'{aggregation.capitalize()} {value_col}',
            height=self.default_height,
            hovermode='x unified'
        )
        
        fig.update_xaxes(rangeslider_visible=True)
        
        return self._apply_theme(fig)
    
    # ==================== COMPOSITION VISUALIZATIONS ====================
    
    def plot_pie_chart(
        self,
        data: pd.DataFrame,
        column: str,
        top_n: Optional[int] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive pie chart
        
        Args:
            data: DataFrame
            column: Column to visualize
            top_n: Show only top N categories
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        value_counts = data[column].value_counts()
        
        if top_n:
            value_counts = value_counts.head(top_n)
        
        title = title or f'Distribution of {column}'
        
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=title,
            hole=0.3  # Make it a donut chart
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}'
        )
        
        fig.update_layout(height=self.default_height)
        
        return self._apply_theme(fig)
    
    def plot_sunburst(
        self,
        data: pd.DataFrame,
        path_columns: List[str],
        value_col: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create hierarchical sunburst chart
        
        Args:
            data: DataFrame
            path_columns: Columns defining hierarchy (from outer to inner)
            value_col: Column for values (if None, uses count)
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        title = title or f'Hierarchical View: {" → ".join(path_columns)}'
        
        if value_col:
            fig = px.sunburst(
                data,
                path=path_columns,
                values=value_col,
                title=title
            )
        else:
            # Use count
            count_data = data.groupby(path_columns).size().reset_index(name='count')
            fig = px.sunburst(
                count_data,
                path=path_columns,
                values='count',
                title=title
            )
        
        fig.update_layout(height=self.default_height)
        
        return self._apply_theme(fig)
    
    def plot_treemap(
        self,
        data: pd.DataFrame,
        path_columns: List[str],
        value_col: Optional[str] = None,
        color_col: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create hierarchical treemap
        
        Args:
            data: DataFrame
            path_columns: Columns defining hierarchy
            value_col: Column for values (if None, uses count)
            color_col: Column for coloring
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        title = title or f'Treemap: {" → ".join(path_columns)}'
        
        if value_col is None:
            # Use count
            count_data = data.groupby(path_columns).size().reset_index(name='count')
            value_col = 'count'
            plot_data = count_data
        else:
            plot_data = data
        
        fig = px.treemap(
            plot_data,
            path=path_columns,
            values=value_col,
            color=color_col,
            title=title
        )
        
        fig.update_layout(height=self.default_height)
        
        return self._apply_theme(fig)
    
    # ==================== ADVANCED VISUALIZATIONS ====================
    
    def plot_parallel_coordinates(
        self,
        data: pd.DataFrame,
        dimensions: List[str],
        color_by: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create parallel coordinates plot for multivariate analysis
        
        Args:
            data: DataFrame
            dimensions: List of columns to include as dimensions
            color_by: Column to color by
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        title = title or 'Parallel Coordinates Plot'
        
        fig = px.parallel_coordinates(
            data,
            dimensions=dimensions,
            color=color_by,
            title=title
        )
        
        fig.update_layout(height=self.default_height)
        
        return self._apply_theme(fig)
    
    def plot_parallel_categories(
        self,
        data: pd.DataFrame,
        dimensions: List[str],
        color_by: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create parallel categories (Sankey-style) plot
        
        Args:
            data: DataFrame
            dimensions: List of categorical columns
            color_by: Column to color by
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        title = title or f'Flow Analysis: {" → ".join(dimensions)}'
        
        fig = px.parallel_categories(
            data,
            dimensions=dimensions,
            color=color_by,
            title=title
        )
        
        fig.update_layout(height=self.default_height)
        
        return self._apply_theme(fig)
    
    def plot_3d_scatter(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        color_by: Optional[str] = None,
        size_by: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create 3D scatter plot
        
        Args:
            data: DataFrame
            x_col: Column for x-axis
            y_col: Column for y-axis
            z_col: Column for z-axis
            color_by: Column for color grouping
            size_by: Column for point size
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        title = title or f'3D Scatter: {x_col}, {y_col}, {z_col}'
        
        fig = px.scatter_3d(
            data,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_by,
            size=size_by,
            title=title,
            hover_data=data.columns
        )
        
        fig.update_layout(
            height=self.default_height,
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            )
        )
        
        return self._apply_theme(fig)
    
    # ==================== DASHBOARD LAYOUTS ====================
    
    def create_overview_dashboard(
        self,
        data: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str],
        target_col: str
    ) -> go.Figure:
        """
        Create comprehensive overview dashboard with multiple subplots
        
        Args:
            data: DataFrame
            numerical_cols: List of numerical columns to visualize
            categorical_cols: List of categorical columns to visualize
            target_col: Target variable column
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Target Distribution',
                'Top Categories',
                'Numerical Feature Distribution',
                'Correlation with Target'
            ),
            specs=[
                [{'type': 'pie'}, {'type': 'bar'}],
                [{'type': 'box'}, {'type': 'bar'}]
            ]
        )
        
        # 1. Target distribution (pie chart)
        target_counts = data[target_col].value_counts()
        fig.add_trace(
            go.Pie(labels=target_counts.index, values=target_counts.values, hole=0.3),
            row=1, col=1
        )
        
        # 2. Top categories (bar chart)
        if categorical_cols:
            cat_col = categorical_cols[0]
            cat_counts = data[cat_col].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=cat_counts.index, y=cat_counts.values, name=cat_col),
                row=1, col=2
            )
        
        # 3. Numerical distribution (box plot)
        if numerical_cols:
            for num_col in numerical_cols[:3]:  # Show first 3
                fig.add_trace(
                    go.Box(y=data[num_col], name=num_col),
                    row=2, col=1
                )
        
        # 4. Correlation with target
        if numerical_cols and pd.api.types.is_numeric_dtype(data[target_col]):
            correlations = data[numerical_cols].corrwith(data[target_col]).sort_values(key=abs, ascending=False).head(10)
            fig.add_trace(
                go.Bar(
                    x=correlations.values,
                    y=correlations.index,
                    orientation='h',
                    marker_color=['red' if x < 0 else 'green' for x in correlations.values]
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text='Data Overview Dashboard',
            title_font_size=20,
            showlegend=True,
            height=800,
            width=1200
        )
        
        return self._apply_theme(fig)
    
    def save_figure(self, fig: go.Figure, filename: str, format: str = 'html'):
        """
        Save plotly figure to file
        
        Args:
            fig: Plotly figure object
            filename: Output filename
            format: Output format ('html', 'png', 'jpg', 'svg', 'pdf')
        """
        if format == 'html':
            fig.write_html(filename)
        else:
            fig.write_image(filename)
        
        print(f"Figure saved to: {filename}")


# Convenience function for quick dashboard creation
def create_quick_dashboard(data: pd.DataFrame, target_col: str) -> go.Figure:
    """
    Quickly create an overview dashboard
    
    Args:
        data: DataFrame
        target_col: Target variable column name
        
    Returns:
        Plotly figure object
    """
    dashboard = InteractiveDashboard()
    
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    return dashboard.create_overview_dashboard(data, numerical_cols, categorical_cols, target_col)
