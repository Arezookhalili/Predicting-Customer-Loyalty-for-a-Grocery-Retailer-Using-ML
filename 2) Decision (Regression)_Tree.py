###############################################################################
# Decision (Regression) Tree - ABC Grocery Task
###############################################################################


###############################################################################
# Import Required Packages
###############################################################################

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

###############################################################################
# Import Sample Data
###############################################################################

# Import
data_for_model = pickle.load(open('Data/abc_regression_modelling.p', 'rb'))                 # 'rb': read file

# Drop unnecessary columns
data_for_model.drop('customer_id', axis = 1, inplace = True)

# Shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

###############################################################################
# Deal with Missing Values
###############################################################################

data_for_model.isna().sum()
data_for_model.dropna(how = 'any', inplace = True)

###############################################################################
# Deal with Outliers: No need to remove them as we act based on the proposed criteria 
###############################################################################

###############################################################################
# Split Input Variables and Output Variables
###############################################################################

X = data_for_model.drop(['customer_loyalty_score'], axis = 1)
y = data_for_model['customer_loyalty_score']

###############################################################################
# Split out Training and Test Sets
###############################################################################

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 42)

###############################################################################
# Deal with Categorical Variables
###############################################################################

# Create a list of categorical variables                    
categorical_vars = ['gender']   

# Create and apply OneHotEncoder while removing the dummy variable
one_hot_encoder = OneHotEncoder(sparse = False, drop = 'first')               

# Apply fit_transform on training data
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])

# Apply transform on test data
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# Get feature names to see what each column in the 'X_train_encoded' presents
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Convert our result from an array to a DataFrame
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)

# Concatenate (Link together in a series or chain) new DataFrame to our original DataFrame 
X_train = pd.concat([X_train.reset_index(drop = True),X_train_encoded.reset_index(drop = True)], axis = 1)    
X_test = pd.concat([X_test.reset_index(drop = True),X_test_encoded.reset_index(drop = True)], axis = 1)    
 
# Drop the original categorical variable columns
X_train.drop(categorical_vars, axis = 1, inplace = True)           
X_test.drop(categorical_vars, axis = 1, inplace = True)           

###############################################################################
# Feature Selection: Here, each variable is judged independently. Also, applying feature selection does not change the accuracy of decision tree as each variable is judged separately but we can use it to decrease the computation cost.
# It is not required but it can be used to increase the computation speed in case we have high number of variables
###############################################################################

###############################################################################
# Model Training
###############################################################################

regressor = DecisionTreeRegressor(random_state = 42, max_depth = 4)
 # regressor = DecisionTreeRegressor(random_state = 42, max_depth = 4)       # To refit the model based on the refit explanation section

regressor.fit(X_train, y_train)

###############################################################################
# Model Assessment
###############################################################################

# Predict on the test set
y_pred = regressor.predict(X_test)

# Calculate R-squared
r_squared = r2_score(y_test, y_pred)
print(r_squared)

# Cross validation (KFold: including both shuffling and the random state)
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)                    # n_splits: number of equally sized chunk of data
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = 'r2')
cv_scores.mean()

# Calculate adjusted R-squared
num_data_points, num_input_vars = X_test.shape                           # R should be calculated using test data as we want to compare y_test and y_pred
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

# A Demonstration of overfitting

y_pred_training = regressor.predict(X_train)        # As we trained our model on X_train, predictin y_train would result in overfitting
r2_score(y_train, y_pred_training)

# Finding the best max depth

max_depth_list = list(range(1,9))
accuracy_scores = []

for depth in max_depth_list:
    
    regressor = DecisionTreeRegressor(max_depth = depth, random_state = 42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    accuracy = r2_score(y_test,y_pred)
    accuracy_scores.append(accuracy)
    
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]

# Plot of max depths
plt.plot(max_depth_list, accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker = 'x', color = 'red')
plt.title(f'Accuracy by Max Depth \n Optimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy, 4)})')
plt.xlabel('Max Depth of Decision Tree')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()

# Refit the model with max depth that gives us much more explainable model with good accuracy
"""
as the accuracy does not change much for depth > 4, we can set max_depth = 4 and run the code again to get a more understandable decision tree)
regressor = DecisionTreeRegressor(random_state = 42, max_depth = 4)       # To refit the model based on the refit explanation section
"""

# Plot our model

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 24)
"""
The variable on top of the tree is the one that is very important in predicting our output.
""" 
