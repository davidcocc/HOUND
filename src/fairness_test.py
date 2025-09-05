import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import os
from model_utils import load_keras_model, predict_file
from utils import resolve_project_root



BASE_DIR = os.path.dirname(__file__)   # cartella in cui sta fairness_test.py
csv_path = os.path.join(BASE_DIR, "UrbanSound8K_fairness.csv")

# Carica il CSV
df = pd.read_csv(csv_path)

# Carica il modello
ROOT = resolve_project_root(__file__)
model = load_keras_model(ROOT)

# Funzione per predire la classe di un file
def predict_class(filepath):
    cls, _, _, _ = predict_file(model, filepath)
    return cls

# Percorso base dei file audio
audio_base = os.path.join(ROOT, "data", "archive")

# Applica la predizione a ogni riga del DataFrame solo se non esiste già la colonna
if 'predicted_classID' not in df.columns:
    def get_full_path(row):
        return os.path.join(audio_base, f"fold{row['fold']}", row['slice_file_name'])
    df['predicted_classID'] = df.apply(lambda row: predict_class(get_full_path(row)), axis=1)
    df.to_csv(csv_path, index=False)  # Salva il CSV con le predizioni

def compute_metrics(group):
    y_true = group['classID']
    y_pred = group['predicted_classID']
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    # FPR one-vs-rest per la classe target
    class_target = group['classID'].iloc[0]
    y_true_bin = (y_true == class_target).astype(int)
    y_pred_bin = (y_pred == class_target).astype(int)
    # Forza le etichette per avere sempre una matrice 2x2
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn + 1e-10)
    return pd.Series({'accuracy': acc, 'f1': f1, 'fpr': fpr})

metrics = df.groupby(['classID', 'salience', 'fold']).apply(compute_metrics).reset_index()

X = 0.10  # 10%

def flag_gaps(metric_df, metric_name, groupby_cols):
    flagged = []
    for name, group in metric_df.groupby(groupby_cols):
        max_val = group[metric_name].max()
        min_val = group[metric_name].min()
        gap = max_val - min_val
        if gap > X:
            flagged.append({'group': name, 'metric': metric_name, 'gap': gap})
    return pd.DataFrame(flagged)

flagged_acc = flag_gaps(metrics, 'accuracy', ['classID', 'fold'])
flagged_f1 = flag_gaps(metrics, 'f1', ['classID', 'fold'])
flagged_fpr = flag_gaps(metrics, 'fpr', ['classID', 'fold'])

print("Accuracy gaps > 10%:")
print(flagged_acc)
print("F1 gaps > 10%:")
print(flagged_f1)
print("FPR gaps > 10%:")
print(flagged_fpr)

# Salva i risultati su CSV
metrics.to_csv(os.path.join(BASE_DIR, "metrics_fairness/fairness_metrics.csv"), index=False)
flagged_acc.to_csv(os.path.join(BASE_DIR, "metrics_fairness/fairness_gaps_accuracy.csv"), index=False)
flagged_f1.to_csv(os.path.join(BASE_DIR, "metrics_fairness/fairness_gaps_f1.csv"), index=False)
flagged_fpr.to_csv(os.path.join(BASE_DIR, "metrics_fairness/fairness_gaps_fpr.csv"), index=False)