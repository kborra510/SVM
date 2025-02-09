import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        
        """
        Fit the SVM model to training data.
        
        Parameters:
        - X (np.array): Training features.
        - y (np.array): Target values.

        Returns:
        - None
        """
        # Initialize weights and bias to zeros
        self.w = np.zeros(X.shape[1])
        self.b = 0

        # Implement gradient descent to update weights and bias
        for _ in range(self.n_iters):
            for index, x in enumerate(X):
                cond = y[index] * (np.dot(x, self.w) - self.b) >= 1
                if cond:
                    weighted_term = self.lr * (2 * self.lambda_param * self.w)
                    self.w -= weighted_term
                else:
                    weighted_term = self.lr * (2 * self.lambda_param * self.w - np.dot(x, y[index]))
                    self.w -= weighted_term
                    self.b -= self.lr * y[index]

    def predict(self, X):

        """
        Predict class labels for samples in X.

        Parameters:
        - X (np.array): Test features.

        Returns:
        - np.array: Predicted class label per sample.
        """
        # Compute the linear combination of weights and features plus bias
        # Return class labels based on the sign of the linear combination
        preds = np.dot(X, self.w) - self.b
        return np.sign(preds)

# Load and preprocess dataset
iris = load_iris()
X = iris.data
y = iris.target
y_class = list(np.unique(y))
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Training the model
# Training multiple binary classifiers using one-vs-all strategy
classifiers = []
for i in range(len(y_class)):  # Three classes in Iris dataset
    svm = LinearSVM()
    y_binary = np.where(y_train == i, 1, -1)  # Convert to binary labels
    svm.fit(X_train, y_binary)
    classifiers.append(svm)
# Make predictions
def predict_one_vs_all(classifiers, X):
    preds = np.zeros((X.shape[0], len(classifiers)))
    for idx, clf in enumerate(classifiers):
        preds[:, idx] = clf.predict(X)
    return np.argmax(preds, axis=1)
# Calculate the metrics using scikit-learn
y_pred = predict_one_vs_all(classifiers, X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_score = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f'Accuracy: {round(accuracy,2)}')
print(f'Precision: {round(precision,2)}')
print(f'Recall: {round(recall,2)}')
print(f'F1 Score: {round(f1_score,2)}')

# Plotting the decision boundary and save the figure.
def plot_decision_boundary(X, y, model, target_names):
    ## creating a boundary of x and y

    x_min, y_min = np.min(X[:, 0]) - 1, np.min(X[:, 1]) - 1  # Min boundary for feature1, feature2
    x_max, y_max = np.max(X[:, 0]) + 1, np.max(X[:, 0]) + 1  # Max boundary for feature1, feature2

    ## forming a grid on which we can draw decision boundary
    x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, 0.05),
                                 np.arange(y_min, y_max, 0.05))
    pred = model(np.array([x_grid.ravel(), y_grid.ravel()]).T)
    pred = pred.reshape(x_grid.shape)
    ## Plotting decision boundary
    plt.contourf(x_grid, y_grid, pred, alpha=0.27)
    for i, name in enumerate(np.unique(y)):
        class_name = target_names[name]  # Map label to class name
        plt.scatter(X[y == name][:, 0], X[y == name][:, 1], label=f'{class_name}', edgecolors='k')
    plt.legend()
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal Width')
    plt.title('SVM (One vs All)')
    plt.savefig('svm.png')
    plt.show()


X_des = iris.data[:, :2]  # Consider only first two features for plotting
y = iris.target
feature_names = iris.feature_names[:2]  # Names of the first two features

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_des, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
classifiers_des = []
for i in range(3):  # Three classes in Iris dataset
    svm = LinearSVM()
    y_binary = np.where(y_train == i, 1, -1)  # Convert to binary labels
    svm.fit(X_train, y_binary)
    classifiers_des.append(svm)
plot_decision_boundary(X_test, y_test, lambda x: predict_one_vs_all(classifiers_des, x), iris.target_names)

