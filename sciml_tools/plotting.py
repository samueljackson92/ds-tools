import numpy as np
import pandas as pd
import seaborn as sns

def plot_cm(cm, norm=True, labels=['True', 'False'], cmap='Blues'):
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    if norm:
        cm = cm / np.sum(cm)

    sns.set(font_scale=1.14)
    sns.heatmap(cm, annot=True, cmap=cmap, fmt='d' if not norm else '.2f')
