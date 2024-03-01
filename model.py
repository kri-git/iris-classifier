from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

labels = {
    0 : "setosa",
    1 : "versicolor",
    2 : "virginica"
}

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6,)

# Create and train decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Save the trained model
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(clf, file)

# Load the saved model
with open('decision_tree_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Make predictions using the loaded model
predictions = loaded_model.predict(X_test)

predicted_labels = [labels[pred] for pred in predictions]
print("\nPredicted Label:", predicted_labels[0])