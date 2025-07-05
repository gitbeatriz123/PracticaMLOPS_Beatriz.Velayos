import mlflow
from functions import load_data, preprocess_text, train_classifier, evaluate_model
from sklearn.model_selection import train_test_split
import argparse

def main(C=1.0, max_iter=1000):
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_vect, X_test_vect, vectorizer = preprocess_text(X_train, X_test)
    clf = train_classifier(X_train_vect, y_train, C=C, max_iter=max_iter)
    metrics = evaluate_model(clf, X_test_vect, y_test, target_names)
    
    # Registro en MLflow
    mlflow.set_experiment("text_classification_practica_final")
    with mlflow.start_run():
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        print("Métricas registradas en MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0, help="Parámetro C del clasificador")
    parser.add_argument('--max_iter', type=int, default=1000, help="Máximo número de iteraciones")
    args = parser.parse_args()
    main(C=args.C, max_iter=args.max_iter)