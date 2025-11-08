from sklearn import tree    # Decision tree for ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from DecisionTree import DecisionTreeClassifier
import numpy as np

def generate_height_weight_shoe_dataset(n_samples=200, seed=42):
    """
    Generate n_samples (features, labels).
    Features: [height_cm, weight_kg, shoe_size_eu]
    Labels: 'male' or 'female'
    """
    np.random.seed(seed)
    # Approximately half male and half female
    n_male = n_samples // 2
    n_female = n_samples - n_male

    # Normal distributions to generate samples
    male_heights = np.random.normal(loc=178, scale=7, size=n_male)   # cm
    male_weights = np.random.normal(loc=82,  scale=10, size=n_male)  # kg
    male_shoes  = np.random.normal(loc=43,  scale=2, size=n_male)   # EU size

    female_heights = np.random.normal(loc=165, scale=6, size=n_female)
    female_weights = np.random.normal(loc=66,  scale=8, size=n_female)
    female_shoes  = np.random.normal(loc=38,  scale=2, size=n_female)

    # Concatenate atributes
    heights = np.concatenate((male_heights, female_heights))
    weights = np.concatenate((male_weights, female_weights))
    shoes   = np.concatenate((male_shoes, female_shoes))

    labels = np.array(['male'] * n_male + ['female'] * n_female)

    # Limit the data to a certain range of atributes
    heights = np.clip(heights, 140, 210).round().astype(int)
    weights = np.clip(weights, 40, 140).round().astype(int)
    shoes   = np.clip(shoes, 34, 50).round().astype(int)

    # Build X and Y
    X = np.column_stack((heights, weights, shoes))
    Y = labels.reshape(-1, 1)

    # Shuffle data
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]

    # Return as lists
    return X.tolist(), Y.flatten().tolist()

# Calculate the data
X, Y = generate_height_weight_shoe_dataset(n_samples=1000, seed=42)

# Variable to store decision tree model
clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

prediction = clf.predict([[190,100,43]])

print(prediction)

# Using the DecisionTree program we made
X, Y = np.array(X, dtype=float), np.array(Y, dtype=object) # Convert again to array
Y = Y.reshape(-1, 1) # In order to have a 2D dimensions array

# Train-Test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

# Create the DecisionTreeClassifier class --> If the number of samples in a
# node is less than min_samples_split, we won't split anymore. If the depth
# of the tree reaches the max_depth, it will stop spliting the tree
classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=5)
classifier.fit(X_train, Y_train)   # Do the fit using the training data
classifier.print_tree()  # Print the tree

# Finally we try to predict
Y_pred = classifier.predict(X_test)
# Accuracy of the model
print(f"\n The accuracy score of this tree is: {accuracy_score(Y_test, Y_pred)}")