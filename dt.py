import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#################
# ID3 Algorithm
#################

def entropy(target_col):
    """
    Calculate the entropy of a dataset.
    """
    # Get the counts of different classes in the target column
    class_counts = np.unique(target_col, return_counts=True)[1]
    # Calculate the total number of instances
    total_instances = len(target_col)
    # Initialize entropy
    entropy = 0
    # For each class, calculate its probability and then its contribution to entropy
    for count in class_counts:
        p = count / total_instances
        entropy -= p * np.log2(p)
    return entropy


def InfoGain(data, split_attribute_idx, target_index=-1):
    """
    Calculate the information gain of a dataset. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_idx = the index of the feature for which the information gain should be calculated
    3. target_idx = the index of the target feature. The default value is "class"
    """
    target_e = entropy(data[:, target_index])
    values = np.unique(data[:, split_attribute_idx])
    # Handling empty splits
    if len(values) == 1:
        return 0
    weighted_e = 0
    for value in values:
        # Filter data for current split
        filter = data[data[:, split_attribute_idx] == value]
        weighted_e += (len(filter) / len(data)) * entropy(filter[:, target_index])

    return target_e - weighted_e



def ID3(data, originaldata, features, target_idx=-1, parent_node_class = None):
    """
    ID3 Algorithm: This function takes five parameters:
    1. data = the dataset for which the ID3 algorithm should be run — in the first run, this should be the entire dataset
    2. originaldata = this is the original dataset needed to calculate the mode target feature value of the original dataset
    in the first run
    3. features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
    we have to remove features from our dataset — slicing the feature space
    4. target_attribute_name = the name of the target attribute
    5. parent_node_class = This is the value or class of the mode target feature value of the parent node for a specific node. This is
    also needed for the recursive call in case the dataset delivered to the ID3 algorithm consists only of samples with the same class. Then,
    this function returns the mode target feature value of the parent node dataset as the class value of the dataset.
    """
    # Start with stop conditions:
    # 1. All samples belong to the same class (pure node)
    if len(np.unique(data[:, target_idx])) == 1:
        return data[0, target_idx]
    # 2. If the dataset is empty, return the most frequent target feature value in the original dataset
    if len(data) == 0:
        return np.unique(originaldata[:, target_idx])[
            np.argmax(np.unique(originaldata[:, target_idx], return_counts=True)[1])]
    # 3. No more features remaining to split on
    if len(features) == 0:
        # Predict the majority class from the parent node or the original data
        if parent_node_class is not None:
            return parent_node_class
        else:
            return np.bincount(data[:, target_idx]).argmax()

    parent_node_class = np.unique(data[:, target_idx])[np.argmax(np.unique(data[:, target_idx], return_counts=True)[1])]

    # the best attribute to split on highest information gain
    best_attribute = features[np.argmax([InfoGain(data, feat, target_idx) for feat in features])]

    # Create a new decision tree node
    dt = {best_attribute: {}}

    # Get unique values of the best attribute
    values = np.unique(data[:, best_attribute])

    # Recursively build subtrees for each value of the best attribute
    for value in values:
        # Filter data for the current split value
        filtered_data = data[data[:, best_attribute] == value]
        remaining_features = features.copy()
        remaining_features.remove(best_attribute)

        # Call ID3 recursively on the filtered data and update the current node's dictionary
        sub_dt = ID3(filtered_data, originaldata, remaining_features, target_idx,
                      parent_node_class)
        dt[best_attribute][value] = sub_dt

    return dt


# Load and prepare the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Define features
features = [0, 1, 2, 3]  # indices of features to consider

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the training set into a format that the ID3 algorithm can use
train_data = np.column_stack((X_train, y_train))

# Train the ID3 model
tree = ID3(train_data, train_data, features, target_idx=-1)
print(tree.keys())


# Predict method for a single instance
def predict(instance, tree, default=1):
    for attribute in list(tree.keys()):
        if instance[attribute] in list(tree[attribute].keys()):
            result = tree[attribute][instance[attribute]]
            if isinstance(result, dict):
                return predict(instance, result, default)
            else:
                return result
        else:
            return default


# Predict on the test set
most_frequent_class = np.argmax(np.bincount(y_train))  # when we are not able to find value in tree we take the prediction as most frequest class
y_pred = [predict(x, tree, default=most_frequent_class) for x in X_test]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_s = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f'Accuracy for ID3: {round(accuracy,2)}')
print(f'Precision for ID3: {round(precision,2)}')
print(f'Recall for ID3: {round(recall,2)}')
print(f'F1 Score for ID3: {round(f1_s,2)}')


#################
# C 4.5 Algorithm
#################

def entropy(target_col):
    """
    Calculate the entropy of a dataset for a given target column.
    """
    # Get the counts of different classes in the target column
    class_counts = np.unique(target_col, return_counts=True)[1]
    # Calculate the total number of instances
    total_instances = len(target_col)
    # Initialize entropy
    entropy = 0
    # For each class, calculate its probability and then its contribution to entropy
    for count in class_counts:
        p = count / total_instances
        entropy -= p * np.log2(p)
    return entropy


def info_gain(data, split_attribute_idx, target_index=-1):
    """
    Calculate the information gain of splitting the dataset on a specific attribute.
    """
    target_e = entropy(data[:, target_index])
    values = np.unique(data[:, split_attribute_idx])
    # Handling empty splits
    if len(values) == 1:
        return 0
    weighted_e = 0
    for value in values:
        # Filter data for current split
        filter = data[data[:, split_attribute_idx] == value]
        weighted_e += (len(filter) / len(data)) * entropy(filter[:, target_index])

    return target_e - weighted_e


def split_info(data, split_attribute_idx):
    vals, counts = np.unique(data[:, split_attribute_idx], return_counts=True)
    # split_info = 0
    # for v in range(len(vals)):
    #     split_info -= (counts[v]/np.sum(counts))*np.log2(counts[v]/np.sum(counts))
    # return split_info
    split_info = -np.sum([(counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(vals))])
    return split_info


def gain_ratio(data, split_attribute_idx, target_idx=-1):
    """
    Calculate the gain ratio for a dataset based on a given attribute, adjusting information gain for split info.
    """
    gain = info_gain(data, split_attribute_idx, target_idx)
    split_i = split_info(data, split_attribute_idx)
    gr = gain / split_i
    return gr


def best_split(data, features, target_idx):
    """
    Determine the best feature to split on in the dataset, based on the highest gain ratio.
    """
    # Calculate the gain ratio for all features and get the feature with max gain ratio
    gain_ratios = []
    for fea in features:
        g = gain_ratio(data, fea, target_idx)
        gain_ratios.append(g)
    best_feature_index = np.argmax(gain_ratios)
    return features[best_feature_index]


def C45(data, features, target_idx=-1, parent_node_class=None):
    """
    Recursively build the decision tree using the C4.5 algorithm.
    """
    # Start with stop conditions:
    # 1. All samples belong to the same class (pure node)
    if len(np.unique(data[:, target_idx])) == 1:
        return data[0, target_idx]
    # 2. If the dataset is empty, return the most frequent target feature value in the original dataset
    if len(data) == 0:
        return np.unique(data[:, target_idx])[
            np.argmax(np.unique(data[:, target_idx], return_counts=True)[1])]
    # 3. No more features remaining to split on
    if len(features) == 0:
        # Predict the majority class from the parent node or the original data
        if parent_node_class is not None:
            return parent_node_class
        else:
            return np.bincount(data[:, target_idx]).argmax()

    # If none of the above is true, build the tree
    # Set the default value for this node --> the most frequent target feature value of the current node
    parent_node_class = np.unique(data[:, target_idx])[np.argmax(np.unique(data[:, target_idx], return_counts=True)[1])]
    # Select the feature which best splits the dataset, this is where C4.5 differs from ID3
    best_feature = best_split(data, features, target_idx)
    # Create the tree structure
    dt = {best_feature: {}}
    # Remove the feature that was just used for splitting
    remaining_features = features.copy()
    remaining_features.remove(best_feature)
    # Grow a branch under the root node for each possible value of the root node feature
    for value in np.unique(data[:, best_feature]):
        sub_data = data[data[:, best_feature] == value]
        sub_dt = C45(sub_data, remaining_features, target_idx, parent_node_class)
        dt[best_feature][value] = sub_dt

    return dt


# Load and preprocess the Iris dataset
def load_and_prepare_data():
    """
    Load the Iris dataset, preprocess it, and split it into training and testing sets.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Define features
    features = [0, 1, 2, 3]  # indices of features to consider

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)

    return features, X_train, X_test, y_train, y_test


# Train the C4.5 model on the Iris dataset
def train_model(X_train, y_train, features):
    """
    Train the C4.5 decision tree model on the training dataset.
    """
    train_data = np.column_stack((X_train, y_train))

    # Train the ID3 model
    tree = C45(train_data, features)
    return tree


# Evaluate the performance of the C4.5 model on the test dataset
def evaluate_model(tree, X_test, y_test, y_train):
    """
    Evaluate the metrics of the C4.5 model on the test dataset.
    """
    most_frequent_class = np.argmax(np.bincount(
        y_train))  # when we are not able to find value in tree we take the prediction as most frequest class
    # using the same predict func written for ID3 algo
    y_pred = [predict(x, tree, default=most_frequent_class) for x in X_test]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_s = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1_s


# Load and prepare the dataset
features, X_train, X_test, y_train, y_test = load_and_prepare_data()

# Train the C4.5 model
c45_tree = train_model(X_train, y_train, features)

# Predict and evaluate the model
accuracy, precision, recall, f1_s = evaluate_model(c45_tree, X_test, y_test, y_train)

print(f'Accuracy for C4.5: {round(accuracy,2)}')
print(f'Precision for C4.5: {round(precision,2)}')
print(f'Recall for C4.5: {round(recall,2)}')
print(f'F1 Score for C4.5: {round(f1_s,2)}')

# From the above metrics for both ID3 and C4.5 we can see there is a slight increase in metrics \\
# for C4.5. This is beacuse C4.5 is an extended version of ID3 because of the best split mechanism