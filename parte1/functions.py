from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

def load_data():
    data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    X = data.data
    y = data.target
    target_names = data.target_names
    print("Total documentos:", len(X))
    print("Número de categorías:", len(target_names))
    print("Primer texto:\n", X[0][:500])
    return X, y, target_names

def preprocess_text(X_train, X_test):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    print("Vectorización completada: 10000 características.")
    return X_train_vect, X_test_vect, vectorizer

def train_classifier(X_train_vect, y_train, C=1.0, max_iter=1000):
    clf = LogisticRegression(C=C, max_iter=max_iter, multi_class='auto', solver='lbfgs')
    clf.fit(X_train_vect, y_train)
    print("Entrenamiento completado.")
    return clf

def evaluate_model(clf, X_test_vect, y_test, target_names):
    y_pred = clf.predict(X_test_vect)
    report = classification_report(y_test, y_pred, target_names=target_names)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print("Informe de clasificación:\n", report)
    print("Accuracy:", accuracy)
    print("Precisión media ponderada:", precision)
    print("Recall medio ponderado:", recall)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }
    return metrics
