import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initializes the LogisticRegression model.

        Parameters:
        - learning_rate (float): The step size at each iteration.
        - n_iterations (int): Number of iterations over the training dataset.

        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None  # To be initialized in fit method
        self.bias = None  # To be initialized in fit method

    def _sigmoid(self, z):
        """
        compute the sigmoid function.

        Parameters:
        - z (np.array): Linear combination of weights and features plus bias.

        Returns:
        - np.array: Sigmoid of z.
        """
        sig = 1 / (1 + np.exp(-z))
        return sig

    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data.

        Parameters:
        - X (np.array): Training features.
        - y (np.array): Target values.
        
        Returns:
        - self: The instance of the model.

        """
        # Initialize weights and bias
        self.weights = []
        self.bias = []

        num_classes = len(np.unique(y))

        for c in range(num_classes):
            y_conv = np.where(y == c, 1, 0)
            w_c = np.zeros(X.shape[1])
            b_c = 0
            # Gradient descent to update weights and bias
            for _ in range(self.n_iterations):
                dot_mul = np.dot(X, w_c) + b_c
                y_pred = self._sigmoid(dot_mul)
                # Implement the Gradient descent method to update the cost function
                dw = (1 / X.shape[0]) * np.dot(X.T, (y_pred - y_conv))
                db = (1 / X.shape[0]) * np.sum(y_pred - y_conv)

                w_c -= self.learning_rate * dw
                b_c -= self.learning_rate * db
            self.weights.append(w_c)
            self.bias.append(b_c)

    def predict_proba(self, X):
        """
        Predict probability estimates for all classes.

        Parameters:
        - X (np.array): Test features.

        Returns:
        - np.array: Probability of the sample for each class in the model.
        """
        predict_prob = []
        for i in range(len(self.weights)):
            dot_mul = np.dot(X, self.weights[i]) + self.bias[i]
            predicted_class = self._sigmoid(dot_mul)
            predict_prob.append(predicted_class)
        return np.array(predict_prob).T


    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        - X (np.array): Test features.
        - threshold (float): Threshold used to convert probabilities into binary output.

        Returns:
        - np.array: Predicted class label per sample.
        """
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)  #Consider the class with maximum probability

    def plot_decision_boundary(self, X, y, target_names):
        ## creating a boundary of x and y

        x_min, y_min = np.min(X[:, 0]) - 1, np.min(X[:, 1]) - 1     # Min boundary for feature1, feature2
        x_max, y_max = np.max(X[:, 0]) + 1, np.max(X[:, 0]) + 1.5     # Max boundary for feature1, feature2

        ## forming a grid on which we can draw decision boundary
        x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, 0.05),
                             np.arange(y_min, y_max, 0.05))
        pred = self.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
        pred = pred.reshape(x_grid.shape)
        ## Plotting decision boundary
        plt.contourf(x_grid, y_grid, pred, alpha=0.27)
        for i, name in enumerate(np.unique(y)):
            class_name = target_names[name]  # Map label to class name
            plt.scatter(X[y == name][:, 0], X[y == name][:, 1], label=f'{class_name}', edgecolors='k')
        plt.legend()
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal Width')
        plt.title('Logistic Regression (One vs All)')
        plt.savefig('lr.png')
        plt.show()


# Load and preprocess dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Training the model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Calculate the metrics using scikit-learn
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_score = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f'Accuracy: {round(accuracy,2)}')
print(f'Precision: {round(precision,2)}')
print(f'Recall: {round(recall,2)}')
print(f'F1 Score: {round(f1_score,2)}')


# Plotting the decision boundary and save the figure.
X_des = iris.data[:, :2]  # Consider only first two features for plotting
y = iris.target
feature_names = iris.feature_names[:2]  # Names of the first two features

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_des, y, test_size=0.2, random_state=2024)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the model
model_des = LogisticRegression()
model_des.fit(X_train, y_train)

model_des.plot_decision_boundary(X_test, y_test, iris.target_names)



