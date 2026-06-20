import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def plot_class_imbalance(df):
    """
    Visualizes the class imbalance of the dataset using a Plotly pie chart.
    
    Args:
        df (pd.DataFrame): The input dataframe containing the 'Class' column.
        
    Returns:
        plotly.graph_objects.Figure: A plotly pie chart figure showing the distribution 
                                     between normal and fraudulent transactions.
    """
    fig = px.pie(df, names='Class', title='Class Imbalance (0: Normal, 1: Fraud)', hole=0.3)
    return fig

def plot_transactions_over_time(df):
    """
    Plots the distribution of transactions over time (in hours) to identify 
    any time-based patterns in fraudulent activities.
    
    Args:
        df (pd.DataFrame): The input dataframe containing 'Time' and 'Class' columns.
        
    Returns:
        plotly.graph_objects.Figure: A plotly histogram showing transactions over time.
    """
    # Ensure Time_Hr is present or create a copy to avoid mutating the original dataframe
    plot_df = df.copy()
    if 'Time_Hr' not in plot_df.columns:
        # Convert seconds to hours for easier interpretation
        plot_df['Time_Hr'] = plot_df['Time'] / 3600
        
    # Overlay histogram to compare Normal vs Fraud distributions side-by-side
    fig = px.histogram(plot_df, x="Time_Hr", color="Class", nbins=48, barmode="overlay", 
                       title="Transactions over Time (Hours)", opacity=0.7)
    return fig

def plot_correlation_heatmap(df):
    """
    Computes and plots the feature correlation matrix using a heatmap.
    
    Args:
        df (pd.DataFrame): The input dataframe containing numeric features.
        
    Returns:
        tuple: A tuple containing the Plotly figure and the computed correlation dataframe.
    """
    # Calculate the Pearson correlation matrix
    corr = df.corr()
    
    # Generate the heatmap
    fig = px.imshow(corr, text_auto=False, aspect="auto", 
                    title="Feature Correlation Heatmap", 
                    color_continuous_scale="RdBu_r")
    fig.update_layout(height=800)
    return fig, corr

def plot_top_correlations(corr):
    """
    Extracts and visualizes the features that are most highly correlated with the target 'Class'.
    
    Args:
        corr (pd.DataFrame): The pre-computed correlation matrix.
        
    Returns:
        plotly.graph_objects.Figure: A plotly bar chart of the top absolute correlations.
    """
    # Get absolute correlations with 'Class', sort descending, and remove 'Class' itself
    corr_target = abs(corr["Class"]).sort_values(ascending=False).drop("Class")
    
    # Plot as a bar chart
    fig = px.bar(x=corr_target.index, y=corr_target.values, title="Absolute Correlation with Class",
                  labels={'x': 'Features', 'y': 'Absolute Correlation'})
    return fig

def plot_outlier_boxplot(df, feature):
    """
    Creates a boxplot for a specific feature, grouped by the target class, 
    to visually inspect outliers and data distribution bounds.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        feature (str): The name of the feature column to plot.
        
    Returns:
        plotly.graph_objects.Figure: A plotly boxplot figure.
    """
    fig = px.box(df, x="Class", y=feature, color="Class", 
                 title=f"Boxplot of {feature} by Class")
    return fig

def plot_distributions(df, feature):
    """
    Plots the Kernel Density Estimate (KDE) distribution of a given feature, 
    separated by the target class (Normal vs Fraud).
    
    Args:
        df (pd.DataFrame): The input dataframe.
        feature (str): The name of the feature column to analyze.
        
    Returns:
        matplotlib.figure.Figure: A matplotlib figure containing the KDE plots.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot KDE for Normal transactions
    sns.kdeplot(df[df['Class'] == 0][feature], label='Class 0 (Normal)', fill=True, ax=ax, alpha=0.5)
    
    # Plot KDE for Fraudulent transactions
    sns.kdeplot(df[df['Class'] == 1][feature], label='Class 1 (Fraud)', fill=True, ax=ax, alpha=0.5)
    
    ax.set_title(f'KDE Plot of {feature} grouped by Class')
    ax.legend()
    return fig

def plot_pca_2d(df):
    """
    Reduces the feature space to 2 dimensions using Principal Component Analysis (PCA) 
    and plots the result in a 2D scatter plot to observe clustering behavior.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        
    Returns:
        plotly.graph_objects.Figure: A plotly scatter plot of the PCA projection.
    """
    # Drop target and time columns before PCA projection
    X = df.drop(['Class', 'Time'], axis=1, errors='ignore')
    
    # Initialize and fit PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    
    # Create a new dataframe with PCA results
    pca_df = pd.DataFrame(data = components, columns = ['PC1', 'PC2'])
    pca_df['Class'] = df['Class'].astype(str).values
    
    # Plot the 2D projection
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Class', 
                     title="2D PCA on Sampled Data", opacity=0.7,
                     color_discrete_map={'0': 'blue', '1': 'red'})
    return fig

def get_amount_stratification(df):
    """
    Categorizes the transaction amounts into logical bins (Micro, Small, Medium, Large) 
    and calculates the fraud rate for each category.
    
    Args:
        df (pd.DataFrame): The input dataframe containing the 'Amount' column.
        
    Returns:
        tuple: A tuple containing the Plotly bar chart figure and a dataframe with fraud rates.
    """
    plot_df = df.copy()
    
    # Define custom bins for transaction amounts
    bins = [0, 50, 200, 1000, np.inf]
    labels = ['Micro (0-50)', 'Small (50-200)', 'Medium (200-1000)', 'Large (1000+)']
    
    # Categorize 'Amount' into the defined bins
    plot_df['Amount_Category'] = pd.cut(plot_df['Amount'], bins=bins, labels=labels, right=False)
    
    # Group by amount category and class to get the count
    strat_df = plot_df.groupby(['Amount_Category', 'Class'], observed=False).size().reset_index(name='Count')
    
    # Create a grouped bar chart using a logarithmic scale to account for severe class imbalance
    fig = px.bar(strat_df, x='Amount_Category', y='Count', color='Class', barmode='group', 
                 title="Transaction Count by Amount Category (Log Scale)")
    fig.update_yaxes(type="log", title="Log(Count)")
    
    # Create a pivot table to calculate the percentage of fraud per amount category
    pivot_df = strat_df.pivot(index='Amount_Category', columns='Class', values='Count').fillna(0)
    pivot_df['Fraud_Rate (%)'] = (pivot_df[1] / (pivot_df[0] + pivot_df[1])) * 100
    
    return fig, pivot_df[['Fraud_Rate (%)']].round(3)
