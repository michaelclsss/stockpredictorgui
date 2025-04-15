import pandas as pd
import matplotlib.pyplot as plt


def plot_predictions(y_test, y_pred, index, ticker):
    """Generate the prediction vs actual price plot."""
    
    y_pred_series = pd.Series(y_pred, index=index)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(index, y_test, label='Actual Prices', linewidth=2)
    ax.plot(index, y_pred, label='Predicted Prices', linestyle='--')
    ax.fill_between(index, y_test, y_pred_series.values, color='gray', alpha=0.2, label='Error')
    ax.set_title(f'{ticker} Price Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    
    return fig