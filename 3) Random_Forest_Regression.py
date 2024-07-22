###############################################################################
# Random Forest for Regression - ABC Grocery Task
###############################################################################


###############################################################################
# Import Required Packages
###############################################################################

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

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

# Get feature names to see what each column in the 'encoder_vars_array' presents
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Convert our result from an array to a DataFrame
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)

# Concatenate (Link together in a series or chain) new DataFrame to our original DataFrame 
X_train = pd.concat([X_train.reset_index(drop = True),X_train_encoded.reset_index(drop = True)], axis = 1)    
 
# Drop the original categorical variable columns
X_train.drop(categorical_vars, axis = 1, inplace = True)           

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop = True),X_test_encoded.reset_index(drop = True)], axis = 1)    
X_test.drop(categorical_vars, axis = 1, inplace = True)           

###############################################################################
# Feature Selection: Applying feature selection does not change the accuracy of decision tree as each variable is judged separately but we can use it to decrease the computation cost.
###############################################################################

###############################################################################
# Model Training
###############################################################################

regressor = RandomForestRegressor(random_state = 42)
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
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)    
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = 'r2')
cv_scores.mean()

# Calculate adjusted R-squared
num_data_points, num_input_vars = X_test.shape                           # R should be calculated using test data as we want to compare y_test and y_pred
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

# Feature importance (tells us the importance of each input variable in the predictive power of our random forest model)

feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis = 1)
feature_importance_summary.columns = ['input_variable', 'feature_importance']
feature_importance_summary.sort_values(by = 'feature_importance', inplace = True)

plt.barh(feature_importance_summary['input_variable'],feature_importance_summary['feature_importance'])        # Horizontal bar plot
plt.title('Feature Importance of Random Forest')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()

# Permutation importance (preferred method)

result = permutation_importance(regressor, X_test, y_test, n_repeats = 10, random_state = 42)      # n_repeats: How many times we want to apply random shuffling on each input variable

permutation_importance = pd.DataFrame(result['importances_mean'])                                  # importances_mean: average of data we got over n_repeats of random shuffling
permutation_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names, permutation_importance], axis = 1)
permutation_importance_summary.columns = ['input_variable', 'permutation_importance']
permutation_importance_summary.sort_values(by = 'permutation_importance', inplace = True)

plt.barh(permutation_importance_summary['input_variable'],permutation_importance_summary['permutation_importance'])        # Horizontal bar plot
plt.title('Permutation Importance of Random Forest')
plt.xlabel('Permutation Importance')
plt.tight_layout()
plt.show()




# Predictions under the hood (How the algorithm works)
y_pred[0]   

    # Calculate y_pred manually following the algorithm used in Random Forest Regression                                     
new_data = [X_test.iloc[0]]                         # find the 1st row of X_test
regressor.estimators_                               # gives us all decision trees used here

predictions = []
tree_count = 0
for tree in regressor.estimators_:
    prediction = tree.predict(new_data)[0]
    predictions.append(prediction)
    tree_count += 1
    
print(predictions)                                  # gives us a list of predictions obtained from each tree

    # Calculate mean of data predicted in each tree
sum(predictions) / tree_count

# Save required codes
pickle.dump(regressor, open('Data/random_forest_regression_model.p', 'wb'))        # We should pass any object that we want to save
pickle.dump(one_hot_encoder, open('Data/random_forest_regression_ohe.p', 'wb'))












