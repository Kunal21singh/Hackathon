from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def compare_models():
    # Load dataset
    data = load_iris()
    X, y = data.data, data.target

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    # Evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        results[name] = accuracy
        print(f"{name} - Accuracy: {accuracy:.2f}, Precision: {test_precision:.2f}")

    # Print results
    print("Model Comparison:")
    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.2f}")

if __name__ == "__main__":
    compare_models()