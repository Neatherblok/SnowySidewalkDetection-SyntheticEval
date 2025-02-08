# Adjusting the plots to reflect the percentage out of 59

import matplotlib.pyplot as plt
import numpy as np


# Function to plot a single confusion matrix

def plot_confusion_matrix(matrix, title, total=59):

    plt.figure(figsize=(6, 5))

    plt.imshow(matrix, cmap='Blues', interpolation='nearest', vmin=0, vmax=total)

    plt.colorbar()

    

    # Add text annotations with percentages

    for (j, k), value in np.ndenumerate(matrix):

        percentage = f"{value / total * 100:.1f}%"

        plt.text(k, j, f'{value}\n({percentage})', ha='center', va='center', color='black')

    

    plt.xticks([0, 1], ['Clear', 'Snow'])

    plt.yticks([0, 1], ['Clear', 'Snow'])

    plt.xlabel('Predicted')

    plt.ylabel('Actual')

    plt.tight_layout()

    return plt

