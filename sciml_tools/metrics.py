from sklearn.metrics import confusion_matrix

def kss_score(y_true, y_pred):
    """Hanssen-Kuiper Skill Score (KSS)

    A metric measuring the skill of a binary classifer.
     - Metric is 1 when a classifier is perfectly accurate
     - Metric is -1 when a classifier has opposite skill
     - Metric is 0 when a classifier has no skill at all (random)

     Args:
        y_true: true class labels as a 1D array
        y_pred: predicted class labels as a 1D array

    Returns:
        the skill score (float)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return (tp*tn - fp*fn) / ((tp+fp)*(fn+tn))
