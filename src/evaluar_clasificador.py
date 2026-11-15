from pathlib import Path
import csv
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# El script vive en src/ y los datos en data/ (se admiten también data/ junto al script o en la carpeta actual)
base = Path(__file__).resolve().parent
ruta = next(
    (c for c in (base.parent / 'data' / 'mensajes_sinteticos.csv',
                 base / 'data' / 'mensajes_sinteticos.csv',
                 Path.cwd() / 'data' / 'mensajes_sinteticos.csv') if c.exists()),
    base.parent / 'data' / 'mensajes_sinteticos.csv',
)
with ruta.open(encoding='utf-8-sig', newline='') as archivo:
    filas = list(csv.DictReader(archivo))

# Formato actual: mensaje,categoria,respuesta. Solo mensaje y categoria entran al modelo.
X = [r['mensaje'] for r in filas]
y = [r['categoria'] for r in filas]
con_respuesta = sum(1 for r in filas if (r.get('respuesta') or '').strip())


modelo = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), strip_accents='unicode')),
    ('regresion_logistica', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
])


pred = cross_val_predict(modelo, X, y, cv=LeaveOneOut())

print('EVALUACIÓN EXPLORATORIA DEL MODELO BASELINE')
print('Ejemplos:', len(filas), '| con respuesta modelo:', con_respuesta)
print('Exactitud (Accuracy):', round(accuracy_score(y, pred), 3))
print('\nReporte de Clasificación:')
print(classification_report(y, pred, zero_division=0))
print('Matriz de Confusión (filas: real, columnas: predicho):')
print(confusion_matrix(y, pred, labels=sorted(set(y))))
print('Orden de etiquetas:', sorted(set(y)))