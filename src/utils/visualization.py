import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams['font.family'] = 'Serif'

labels = ['Original', 'FE', 'Original + DT', 'FE + DT']
# Helper
def get_metric(d, key):
    if key in d:
        return d[key]
    if f"val_{key}" in d:
        return d[f"val_{key}"]
    if f"test_{key}" in d:
        return d[f"test_{key}"]
    raise KeyError(f"Key '{key}' not found in {list(d.keys())}")

def plot_model_metrics(model_name,
                       val_metric_dicts,   # list 4 dict: [val, val_fe, val_dt, val_fe_dt]
                       test_metric_dicts,  # list 4 dict: [test, test_fe, test_dt, test_fe_dt]
                       metric='mae'):

    metric_name = metric.upper() if metric != 'r2' else 'R²'

    # Lấy điểm số
    val_scores = [get_metric(d, metric) for d in val_metric_dicts]
    test_scores = [get_metric(d, metric) for d in test_metric_dicts]

    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7, 5))

    rects1 = ax.bar(x - width/2, val_scores, width,
                    label='Validation', color='tab:blue',
                    edgecolor='black', linewidth=1.2)
    rects2 = ax.bar(x + width/2, test_scores, width,
                    label='Test', color='tab:red',
                    edgecolor='black', linewidth=1.2)

    ax.set_ylabel(f'{metric_name} (°C)' if metric != 'r2' else metric_name)
    ax.set_title(f'{model_name} — {metric_name}', fontsize=16)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(ncol=2, loc='upper center')

    # Giới hạn trục y
    if metric != 'r2':
        ymax = max(val_scores + test_scores) * 1.15
        ax.set_ylim(0, ymax)
    else:
        ymin = min(val_scores + test_scores) - 0.05
        ymax = max(val_scores + test_scores) + 0.05
        ax.set_ylim(ymin, ymax)

    # Ghi số lên cột
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}' if metric == 'r2' else f'{h:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, h),
                        ha='center', va='bottom', fontsize=9)
    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

def plot_predictions_vs_actual(y_true, y_pred, model_name="Model", figsize=(8, 6)):
    """
    Plot predictions vs actual values with regression line and R² score
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like  
        Predicted values
    model_name : str
        Name of the model for title
    figsize : tuple
        Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    # Add R² to plot
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
            transform=ax.transAxes, fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{model_name} - Predictions vs Actual')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax