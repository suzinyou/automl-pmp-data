import scipy as sp
import numpy as np

def f1_metric(solution, prediction):
    fn = sum(np.multiply(solution, (1 - prediction)))
    tp = sum(np.multiply(solution, prediction))
    fp = sum(np.multiply((1 - solution), prediction))

    eps = 1e-15
    true_pos_num = sp.maximum(eps, tp + fn)
    found_pos_num = sp.maximum(eps, tp + fp)
    tp = sp.maximum(eps, tp)
    tpr = tp / true_pos_num  # true positive rate (recall)
    ppv = tp / found_pos_num  # positive predictive value (precision)
    arithmetic_mean = 0.5 * sp.maximum(eps, tpr + ppv)
    # Harmonic mean:
    f1 = tpr * ppv / arithmetic_mean
    # Average over all classes
    f1 = np.mean(f1)

    base_f1 = 0.5

    score = (f1 - base_f1) / sp.maximum(eps, (1 - base_f1))

    return score

def acc_metric(solution, prediction):
    tn = float(sum(np.multiply((1-prediction), (1-solution))))
    fn = float(sum(np.multiply(solution, (1 - prediction))))
    tp = float(sum(np.multiply(solution, prediction)))
    fp = float(sum(np.multiply((1 - solution), prediction)))

    score = (tp + fn) / (tn + fn + tp + fp)
    return score