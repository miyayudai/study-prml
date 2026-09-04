import matplotlib.pyplot as plt
import os

def save_plot(fig, save_dir, filename, close=False):
    """
    Saves a matplotlib figure to the specified directory.
    Creates the directory if it does not exist.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, bbox_inches='tight')
    if close:
        plt.close(fig)
    print(f"Plot saved to: {filepath}")

def setup_style():
    """
    Sets up common styling for matplotlib plots.
    """
    # Use standard style to avoid missing style errors
    plt.style.use('default')
    
    # Custom common configurations
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (8, 6)
    })
