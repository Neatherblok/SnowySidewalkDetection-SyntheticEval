# Adjusting the plots to reflect the percentage out of 59

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


# Function to plot a single confusion matrix

def plot_confusion_matrix(matrix, title, total=59):
    """
    Plot a single confusion matrix with improved formatting for research papers.
    
    Args:
        matrix (np.ndarray): 2x2 confusion matrix
        title (str): Title for the plot
        total (int): Total number of samples for percentage calculation
    """
    # Set style for better appearance
    plt.style.use('seaborn-v0_8')  # Using a valid seaborn style
    
    # Create figure with appropriate size
    plt.figure(figsize=(8, 6))
    
    # Create heatmap with better color scheme
    sns.heatmap(matrix, annot=False, cmap='Blues', fmt='d', 
                cbar=True, square=True, linewidths=0.5, 
                linecolor='black', vmin=0, vmax=total)
    
    # Add text annotations with both counts and percentages
    for (j, k), value in np.ndenumerate(matrix):
        percentage = f"{value / total * 100:.1f}%"
        plt.text(k + 0.5, j + 0.5, f'{value}\n({percentage})', 
                ha='center', va='center', color='black', fontsize=12)
    
    # Customize ticks and labels
    plt.xticks([0.5, 1.5], ['Clear', 'Snow'], fontsize=12)
    plt.yticks([0.5, 1.5], ['Clear', 'Snow'], fontsize=12, rotation=0)
    
    # Add labels and title
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    plt.ylabel('True Label', fontsize=14, labelpad=10)
    plt.title(title, fontsize=16, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt

def plot_confusion_matrices(confusion_matrices, titles, model_names, total_samples=None, base_save_dir='confusion_matrices'):
    """
    Plot multiple confusion matrices in separate figures, with 3 matrices per figure,
    organized by model.
    
    Args:
        confusion_matrices (list): List of confusion matrices to plot
        titles (list): List of titles for each matrix
        model_names (list): List of model names corresponding to the matrices
        total_samples (int, optional): Total number of samples for percentage calculation
        base_save_dir (str): Base directory to save the figures
    """
    n_matrices = len(confusion_matrices)
    ncols = 3  # Fixed to three columns per figure
    
    # Create base save directory
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Group matrices by model
    model_groups = {}
    for i, model_name in enumerate(model_names):
        if model_name not in model_groups:
            model_groups[model_name] = []
        model_groups[model_name].append((confusion_matrices[i], titles[i]))
    
    # Process each model's matrices
    figures = []
    for model_name, matrices_titles in model_groups.items():
        # Create model-specific directory
        model_dir = os.path.join(base_save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Process matrices in groups of 3
        for i in range(0, len(matrices_titles), ncols):
            # Get the current group of matrices and titles
            current_group = matrices_titles[i:i+ncols]
            current_matrices = [m for m, _ in current_group]
            current_titles = [t for _, t in current_group]
            
            # Create figure for this group
            fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5))
            if ncols == 1:
                axes = [axes]
            
            # Plot each confusion matrix in the current group
            for j, (cm, title) in enumerate(zip(current_matrices, current_titles)):
                ax = axes[j]
                
                # Create heatmap without annotations first
                sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                           cbar_kws={'label': 'Count'}, vmin=0, vmax=total_samples)
                
                # Add text annotations with improved visibility
                for (k, l), z in np.ndenumerate(cm):
                    # Calculate percentage
                    percentage = (z / total_samples) * 100 if total_samples else (z / np.sum(cm)) * 100
                    
                    # Determine text color based on background darkness
                    val = cm[k, l] / total_samples if total_samples else cm[k, l] / np.sum(cm)
                    text_color = 'white' if val > 0.5 else 'black'
                    
                    # Format text with count and percentage
                    text = f'{z}\n({percentage:.1f}%)'
                    
                    # Add text with white outline for better visibility
                    ax.text(l + 0.5, k + 0.5, text,
                           ha='center', va='center',
                           color=text_color,
                           fontsize=10,
                           fontweight='bold',
                           bbox=dict(facecolor='none', edgecolor='none', pad=1))
                
                ax.set_title(title, fontsize=12, pad=10)
                ax.set_xlabel('Predicted', fontsize=10)
                ax.set_ylabel('True', fontsize=10)
                ax.set_xticklabels(['Clear', 'Snow'], fontsize=9)
                ax.set_yticklabels(['Clear', 'Snow'], fontsize=9, rotation=0)
            
            # Hide any unused subplots in the current group
            for j in range(len(current_matrices), ncols):
                axes[j].axis('off')
            
            plt.tight_layout()
            figures.append(fig)
            
            # Save the figure
            row_num = i // ncols + 1
            fig.savefig(os.path.join(model_dir, f'confusion_matrices_row_{row_num}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    return figures

