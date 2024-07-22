###############################################################################
# Linear Regression - ABC Grocery Task
###############################################################################


###############################################################################
# Import Required Packages
###############################################################################

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

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
# Deal with Outliers
###############################################################################

# Describe the data and compare mean, max and min to find the columns that possibly have outliers
outlier_investigation = data_for_model.describe()

outlier_columns = ['distance_from_store', 'total_sales', 'total_items']

# Boxplot approach
for column in outlier_columns:
    
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2                                         # We used 2 instead of 1.5 to keep more data and remove less outliers
    max_border = upper_quartile + iqr_extended
    min_border = lower_quartile - iqr_extended
    outliers = data_for_model[(data_for_model[column] > max_border) | (data_for_model[column] < min_border)].index     # .index: to get the index of outliers to be used in drop command
    print(f'{len(outliers)} outliers were detected in column {column}')
    
    data_for_model.drop(outliers, inplace = True)

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
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])            # we want our model to learn from train data and be applied on test data (not to learn from test data)

# Get feature names to see what each column in the 'encoder_vars_array' presents
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Convert our result from an array to a DataFrame
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)

# Concatenate (Link together in a series or chain) new DataFrame to our original DataFrame 
X_train = pd.concat([X_train.reset_index(drop = True), X_train_encoded.reset_index(drop = True)], axis = 1)    
X_test = pd.concat([X_test.reset_index(drop = True), X_test_encoded.reset_index(drop = True)], axis = 1)    
 
# Drop the original categorical variable columns
X_train.drop(categorical_vars, axis = 1, inplace = True)           
X_test.drop(categorical_vars, axis = 1, inplace = True)           

###############################################################################
# Feature Selection
###############################################################################

regressor = LinearRegression()
feature_selector = RFECV(regressor)                          # We can determine number of chunks (default:5 meaning that it splits the data to 5 equal size chunks, runs the model over 4 chunks and validate it over the remaining one)

fit = feature_selector.fit(X_train,y_train)

# Finding the optimum number of variables
optimal_feature_count = feature_selector.n_features_
print(f'optimal number of features: {optimal_feature_count}')

# Dynamically updating X DataFrame to contain only the new variables
X_train = X_train.loc[:,feature_selector.get_support()]
X_test = X_test.loc[:,feature_selector.get_support()]

# Visualizing the results in case required
plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = 'o')
plt.ylabel('Model Score')
plt.xlabel('Number of Features')
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
"""
\n: goes to next line
round(), 4: round to 4 decimal places
"""
plt.tight_layout()
plt.show()

###############################################################################
# Model Training
###############################################################################

regressor = LinearRegression()
regressor.fit(X_train, y_train)

###############################################################################
# Prediction
###############################################################################

# Predict on the test set
y_pred = regressor.predict(X_test)

###############################################################################
# Model Assessment (Validation)
###############################################################################

# First approach: Calculate R-squared
r_squared = r2_score(y_test, y_pred)
print(r_squared)

# Second approach: Cross validation and Adjusted R-squared
# Cross validation (KFold: including both shuffling and the random state)
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)    
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = 'r2')     # returns r2 for each chunk of data (each cv)
cv_scores.mean()

# Calculate adjusted R-squared
num_data_points, num_input_vars = X_test.shape                           # R should be calculated using test data as we want to compare y_test and y_pred
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

# Extract model coefficients
coefficients = pd.DataFrame(regressor.coef_)
input_variable_names = pd.DataFrame(X_train.columns)
summary_stats = pd.concat([input_variable_names, coefficients], axis = 1)
summary_stats.columns = ['input_variable', 'coefficient']

# Extract model intercept
regressor.intercept_

