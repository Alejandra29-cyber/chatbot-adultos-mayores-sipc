from pathlib import Path
import csv
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ruta = Path(__file__).parent / 'data' / 'mensajes_sinteticos.csv'
with ruta.open(encoding='utf-8', newline='') as archivo:
    filas = list(csv.DictReader(archivo))

X = [r['mensaje'] for r in filas]
y = [r['categoria'] for r in filas]


modelo = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), strip_accents='unicode')),
    ('regresion_logistica', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
])


pred = cross_val_predict(modelo, X, y, cv=LeaveOneOut())

print('EVALUACIÓN EXPLORATORIA DEL MODELO BASELINE')
print('Exactitud (Accuracy):', round(accuracy_score(y, pred), 3))
print('\nReporte de Clasificación:')
print(classification_report(y, pred, zero_division=0))
print('Matriz de Confusión (filas: real, columnas: predicho):')
print(confusion_matrix(y, pred, labels=sorted(set(y))))
print('Orden de etiquetas:', sorted(set(y)))
