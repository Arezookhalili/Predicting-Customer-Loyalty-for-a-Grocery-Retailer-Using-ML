---
layout: post
title: Predicting Customer Loyalty for a Grocery Retailer Using ML
image: "/posts/regression-title-img.png"
tags: [Customer Loyalty, Machine Learning, Regression, Python]
---

I was asked to append market-level customer loyalty information of a grocery retailer to its database. 

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Key Definition](#overview-definition)
- [01. Data Overview](#data-overview)
- [02. Modelling Overview](#modelling-overview)
- [03. Linear Regression](#linreg-title)
- [04. Decision Tree](#regtree-title)
- [05. Random Forest](#rf-title)
- [06. Modelling Summary](#modelling-summary)
- [07. Predicting Missing Loyalty Scores](#modelling-predictions)
- [08. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

The overall aim of this work was to accurately predict the *loyalty score* for those customers who could not be tagged, enabling my client a clear understanding of true customer loyalty, regardless of total spend volume - and allowing for more accurate and relevant customer tracking, targeting, and comms.

To achieve this, I built a predictive model to find relationships between customer metrics and *loyalty score* for those customers who were tagged, and used this to predict the loyalty score metric for those who were not.
<br>
<br>
### Actions <a name="overview-actions"></a>

I first needed to compile the necessary data from tables in the database, gathering key customer metrics that might help predict *loyalty score*, appending on the dependent variable, and separating those who did and did not have this dependent variable present.

As I was predicting a numeric output, I tested three regression modeling approaches, namely:

* Linear Regression
* Decision Tree
* Random Forest
<br>
<br>

### Results <a name="overview-results"></a>

The Random Forest had the highest predictive accuracy.

<br>
**Metric 1: Adjusted R-Squared (Test Set)**

* Random Forest = 0.955
* Decision Tree = 0.886
* Linear Regression = 0.754

<br>
**Metric 2: R-Squared (K-Fold Cross Validation, k = 4)**

* Random Forest = 0.925
* Decision Tree = 0.871
* Linear Regression = 0.853

As the most important outcome for this project was predictive accuracy, rather than explicitly understanding weighted drivers of prediction, I chose the Random Forest as the model to use for making predictions on the customers who were missing the *loyalty score* metric.
<br>
<br>

### Key Definition  <a name="overview-definition"></a>

The *loyalty score* metric measured the % of grocery spend (market level) that each customer allocated to the client vs. all of the competitors.  

Example 1: Customer X had a total grocery spend of $100 and all of this was spent with my client. Customer X had a *loyalty score* of 1.0

Example 2: Customer Y had a total grocery spend of $200 but only 20% is spent with my client. The remaining 80% was spent with competitors. Customer Y had a *customer loyalty score* of 0.2
<br>
<br>
___

# Data Overview  <a name="data-overview"></a>

I was supposed to predict the *loyalty_score* metric. This metric existed (for half of the customer base) in the *loyalty_scores* table of the client database.

The *transactions* table, the *customer_details* table, and the *product_areas* table were used to predict the missing loyalty scores.

Using pandas in Python, I merged *transactions* table, the *customer_details* table, and the *product_areas* table for all customers, creating a single dataset to be used for predicting the missing loyalty scores.

```python
# import required packages
import pandas as pd
import pickle

# import required data tables
loyalty_scores = ...
customer_details = ...
transactions = ...

# merge loyalty score data and customer details data, at customer level
data_for_regression = pd.merge(customer_details, loyalty_scores, how = "left", on = "customer_id")

# aggregate sales data from transactions table
sales_summary = transactions.groupby("customer_id").agg({"sales_cost" : "sum",
                                                         "num_items" : "sum",
                                                         "transaction_id" : "nunique",
                                                         "product_area_id" : "nunique"}).reset_index()

# rename columns for clarity
sales_summary.columns = ["customer_id", "total_sales", "total_items", "transaction_count", "product_area_count"]

# engineer an average basket value column for each customer
sales_summary["average_basket_value"] = sales_summary["total_sales"] / sales_summary["transaction_count"]

# merge the sales summary with the overall customer data
data_for_regression = pd.merge(data_for_regression, sales_summary, how = "inner", on = "customer_id")

# training set: split out data for modeling (loyalty score is present)
regression_modelling = data_for_regression.loc[data_for_regression["customer_loyalty_score"].notna()]

# test set: split out data for scoring post-modeling (loyalty score is missing)
regression_scoring = data_for_regression.loc[data_for_regression["customer_loyalty_score"].isna()]

# for the scoring set, drop the loyalty score column (as it is blank/redundant)
regression_scoring.drop(["customer_loyalty_score"], axis = 1, inplace = True)

# save the datasets for future use
pickle.dump(regression_modelling, open("data/customer_loyalty_modelling.p", "wb"))
pickle.dump(regression_scoring, open("data/customer_loyalty_scoring.p", "wb"))
```
<br>
After this data pre-processing in Python, I had a dataset for modeling containing the following fields...

| **Variable Name** | **Variable Type** | **Description** |
|---|---|---|
| loyalty_score | Dependent | The % of total grocery spend that each customer allocates to ABC Grocery vs. competitors |
| distance_from_store | Independent | "The distance in miles from the customer's home address, and the store" |
| gender | Independent | The gender provided by the customer |
| credit_score | Independent | The customer's most recent credit score |
| total_sales | Independent | Total spend by the customer in ABC Grocery within the latest 6 months |
| total_items | Independent | Total products purchased by the customer in ABC Grocery within the latest 6 months |
| transaction_count | Independent | Total unique transactions made by the customer in ABC Grocery within the latest 6 months |
| product_area_count | Independent | The number of product areas within ABC Grocery the customers has shopped into within the latest 6 months |
| average_basket_value | Independent | The average spend per transaction for the customer in ABC Grocery within the latest 6 months |

___
<br>
# Modelling Overview

I built a model to accurately predict the “loyalty_score” metric for those customers that were able to be tagged, based upon the customer metrics listed above.

If this model works well, I can use this model to predict the customer loyalty score for the customers who were unable to be tagged by the agency.

As I was predicting a numeric output, I tested three regression modeling approaches, namely:

* Linear Regression
* Decision Tree
* Random Forest

___
<br>
# Linear Regression <a name="linreg-title"></a>

I utilized the scikit-learn library within Python to model the data using Linear Regression. The code sections below were broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>
### Data Import <a name="linreg-import"></a>

I imported my modeling data that was saved as a pickle file. I then removed the id column and shuffle our data.
```python
# import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

# import modeling data
data_for_model = pickle.load(open("data/customer_loyalty_modelling.p", "rb"))

# drop unnecessary columns
data_for_model.drop("customer_id", axis = 1, inplace = True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)
```
<br>
### Data Preprocessing <a name="linreg-preprocessing"></a>

For Linear Regression, I had certain data preprocessing steps that needed to be addressed, including:

* Missing values in the data
* The effect of outliers
* Encoding categorical variables to numeric form
* Multicollinearity & Feature Selection

<br>
##### Missing Values

The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value), I just removed those rows.

```python
# remove rows where values were missing
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)
```

<br>
##### Outliers

The ability of a Linear Regression model to generalize well across *all* data could be hampered if there were outliers present. There was no right or wrong way to deal with outliers, but it was always something worth very careful consideration - just because a value was high or low, did not necessarily mean it should not be there!

In this code section, I used **.describe()** from Pandas to investigate the spread of values for each of the predictors. The results of this can be seen in the table below.

| **metric** | **distance_from_store** | **credit_score** | **total_sales** | **total_items** | **transaction_count** | **product_area_count** | **average_basket_value** |
|---|---|---|---|---|---|---|---|
| mean | 2.02 | 0.60 | 1846.50 | 278.30 | 44.93 | 4.31 | 36.78 |
| std | 2.57 | 0.10 | 1767.83 | 214.24 | 21.25 | 0.73 | 19.34 |
| min | 0.00 | 0.26 | 45.95 | 10.00 | 4.00 | 2.00 | 9.34 |
| 25% | 0.71 | 0.53 | 942.07 | 201.00 | 41.00 | 4.00 | 22.41 |
| 50% | 1.65 | 0.59 | 1471.49 | 258.50 | 50.00 | 4.00 | 30.37 |
| 75% | 2.91 | 0.66 | 2104.73 | 318.50 | 53.00 | 5.00 | 47.21 |
| max | 44.37 | 0.88 | 9878.76 | 1187.00 | 109.00 | 5.00 | 102.34 |

<br>
Based on this investigation, I saw some *max* column values for several variables to be much higher than the *median* value.

This was for columns *distance_from_store*, *total_sales*, and *total_items*

For example, the median *distance_to_store* was 1.645 miles, but the maximum was over 44 miles!

Because of this, I applied some outlier removal to facilitate generalization across the full dataset.

I did this using the "boxplot approach" where I removed any rows where the values within those columns were outside of the interquartile range multiplied by 2.

<br>
```python
outlier_investigation = data_for_model.describe()
outlier_columns = ["distance_from_store", "total_sales", "total_items"]

# boxplot approach
for column in outlier_columns:
    
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace = True)
```

<br>
##### Split Out Data For Modelling

In the next code block, we do two things, we first split our data into an **X** object which contains only the predictor variables, and a **y** object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training.  In this case, we have allocated 80% of the data for training, and the remaining 20% for validation.

```python
# split data into X and y objects for modelling
X = data_for_model.drop(["customer_loyalty_score"], axis = 1)
y = data_for_model["customer_loyalty_score"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```

<br>
##### Categorical Predictor Variables

In my dataset, I had one categorical variable *gender* which had values of "M" for Male, "F" for Female, and "U" for Unknown.

As the Linear Regression algorithm could only deal with numerical data, I had to encode my categorical variable.

As *gender* didn't have any explicit *order* to it, in other words, Male wasn't higher or lower than Female and vice versa - one appropriate approach was to apply One Hot Encoding to the categorical column.

One Hot Encoding could be thought of as a way to represent categorical variables as binary vectors, in other words, a set of *new* columns for each categorical value with either a 1 or a 0 saying whether that value was true or not for that observation. These new columns would go into our model as input variables, and the original column was discarded.

I also dropped one of the new columns using the parameter *drop = "first"* to avoid the *dummy variable trap* where my newly created encoded columns perfectly predict each other - and I ran the risk of breaking the assumption that there was no multicollinearity, a requirement or at least an important consideration for some models, Linear Regression being one of them! Multicollinearity occurs when two or more input variables are *highly* correlated with each other, it was a scenario I attempted to avoid in short, while it won't necessarily affect the predictive accuracy of my model, it could make it difficult to trust the statistics around how well the model was performing, and how much each input variable was truly having.

For ease, after I applied One Hot Encoding, I turned my training and test objects back into Pandas Dataframes, with the column names applied.

<br>
```python
# list of categorical variables that need encoding
categorical_vars = ["gender"]

# instantiate OHE class
one_hot_encoder = OneHotEncoder(sparse=False, drop = "first")

# apply OHE
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# extract feature names for encoded columns
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# turn objects back to pandas dataframe
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)
```

<br>
##### Feature Selection

Feature Selection is the process used to select the input variables that are most important to your Machine Learning task.  It can be a very important addition or at least, a consideration, in certain scenarios.  The potential benefits of Feature Selection are:

* **Improved Model Accuracy** - eliminating noise can help true relationships stand out
* **Lower Computational Cost** - our model becomes faster to train and faster to make predictions
* **Explainability** - understanding & explaining outputs for stakeholder & customers becomes much easier

There are many, many ways to apply Feature Selection. These range from simple methods such as a *Correlation Matrix* showing variable relationships, to *Univariate Testing* which helps us understand statistical relationships between variables, and then to even more powerful approaches like *Recursive Feature Elimination (RFE)* which is an approach that starts with all input variables, and then iteratively removes those with the weakest relationships with the output variable.

For this task, I applied a variation of Recursive Feature Elimination called *Recursive Feature Elimination With Cross Validation (RFECV)* where I splitteded the data into many "chunks" and iteratively train & validated models on each "chunk" separately. This means that each time I assessed different models with different variables included, or eliminated, the algorithm also knew how accurate each of those models was. From the suite of model scenarios that were created, the algorithm could determine which provided the best accuracy, and thus could infer the best set of input variables to use!

<br>
```python
# instantiate RFECV & the model type to be utilized
regressor = LinearRegression()
feature_selector = RFECV(regressor)

# fit RFECV onto our training & test data
fit = feature_selector.fit(X_train,y_train)

# extract & print the optimal number of features
optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features: {optimal_feature_count}")

# limit our training & test sets to only include the selected variables
X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]
```

<br>
The below code then produced a plot that visualized the cross-validated accuracy with each potential number of features.

```python
plt.style.use('seaborn-poster')
plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()
```

<br>
This created the below plot, which showed me that the highest cross-validated accuracy (0.8635) was actually when I included all eight of my original input variables. This was marginally higher than 6 included variables, and 7 included variables. I continued on with all 8!

![alt text](/img/posts/lin-reg-feature-selection-plot.png "Linear Regression Feature Selection Plot")

<br>
### Model Training <a name="linreg-model-training"></a>

Instantiating and training the Linear Regression model was done using the below code:

```python
# instantiate my model object
regressor = LinearRegression()

# fit my model using our training & test sets
regressor.fit(X_train, y_train)
```

<br>
### Model Performance Assessment <a name="linreg-model-assessment"></a>

##### Predict On The Test Set

To assess how well my model was predicting new data - I used the trained model object (here called *regressor*) and asked it to predict the *loyalty_score* variable for the test set.

```python
# predict on the test set
y_pred = regressor.predict(X_test)
```

<br>
##### Calculate R-Squared

R-Squared is a metric that shows the percentage of variance in the output variable *y* that is being explained by the input variable(s) *x*.  It is a value that ranges between 0 and 1, with a higher value showing a higher level of explained variance. Another way of explaining this would be to say that, if we had an r-squared score of 0.8 it would suggest that 80% of the variation of our output variable is being explained by our input variables - and something else, or some other variables must account for the other 20%

To calculate r-squared, I used the following code where I passed in my *predicted* outputs for the test set (y_pred), as well as the *actual* outputs for the test set (y_test)

```python
# calculate r-squared for our test set predictions
r_squared = r2_score(y_test, y_pred)
print(r_squared)
```

The resulting r-squared score from this was **0.78**.

<br>
##### Calculate Cross Validated R-Squared

An even more powerful and reliable way to assess model performance is to utilize Cross Validation.

Instead of simply dividing our data into a single training set, and a single test set, with Cross Validation we can break our data into several "chunks" and then iteratively train the model on all but one of the "chunks", test the model on the remaining "chunk" until each has had a chance to be the test set.

The result of this is that we are provided a number of test set validation results - and we can take the average of these to give a much more robust & reliable view of how our model will perform on new, unseen data!

In the code below, I put this into place. I first specified that I wanted 4 "chunks" and then I passed in my regressor object, training set, and test set. I also specified the metric I wanted to assess with, in this case, I sticked to r-squared.

Finally, I tookk a mean of all four test set results.

```python
# calculate the mean cross-validated r-squared for our test set predictions
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()
```

The mean cross-validated r-squared score from this was **0.853**.

<br>
##### Calculate Adjusted R-Squared

When applying Linear Regression with *multiple* input variables, the r-squared metric on it's own *can* end up being an overinflated view of goodness of fit. This is because each input variable will have an *additive* effect on the overall r-squared score. In other words, every input variable added to the model *increases* the r-squared value, and *never decreases* it, even if the relationship is by chance.  

**Adjusted R-Squared** is a metric that compensates for the addition of input variables, and only increases if the variable improves the model above what would be obtained by probability. It is best practice to use Adjusted R-Squared when assessing the results of a Linear Regression with multiple input variables, as it gives a fairer perception of the fit of the data.

```python
# calculate adjusted r-squared for our test set predictions
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)
```

The resulting *adjusted* r-squared score from this was **0.754** which as expected, was slightly lower than the score I got for r-squared on it's own.

<br>
### Model Summary Statistics <a name="linreg-model-summary"></a>

Although my overall goal for this project was predictive accuracy, rather than an explicit understanding of the relationships of each of the input variables and the output variable, it was always interesting to look at the summary statistics for these.
<br>
```python
# extract model coefficients
coefficients = pd.DataFrame(regressor.coef_)
input_variable_names = pd.DataFrame(X_train.columns)
summary_stats = pd.concat([input_variable_names,coefficients], axis = 1)
summary_stats.columns = ["input_variable", "coefficient"]

# extract model intercept
regressor.intercept_
```
<br>
The information from that code block can be found in the table below:

| **input_variable** | **coefficient** |
|---|---|
| intercept | 0.516 |
| distance_from_store | -0.201 |
| credit_score | -0.028 |
| total_sales | 0.000 |
| total_items | 0.001 |
| transaction_count | -0.005 |
| product_area_count | 0.062 |
| average_basket_value | -0.004 |
| gender_M | -0.013 |

<br>
The coefficient value for each of the input variables, along with that of the intercept would make up the equation for the line of best fit for this particular model (or more accurately, in this case, it would be the plane of best fit, as I had multiple input variables).

For each input variable, the coefficient value I saw above tells us, with *everything else staying constant* how many units the output variable (loyalty score) would change with a *one-unit change* in this particular input variable.

To provide an example of this - in the table above, I could see that the *distance_from_store* input variable had a coefficient value of -0.201. This was saying that *loyalty_score* decreased by 0.201 (or 20% as loyalty score was a percentage or at least a decimal value between 0 and 1) for *every additional mile* that a customer lived from the store. This made intuitive sense, as customers who lived a long way from this store, most likely lived near *another* store where they might do some of their shopping as well, whereas customers who lived near this store, probably did a greater proportion of their shopping at this store...and hence had a higher loyalty score!

___
<br>
# Decision Tree <a name="regtree-title"></a>

I again utilized the scikit-learn library within Python to model my data using a Decision Tree. The code sections below were broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>
### Data Import <a name="regtree-import"></a>

I imported my msaved model, removed the id column, and shuffled the data.

```python
# import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

# import modelling data
data_for_model = pickle.load(open("data/customer_loyalty_modelling.p", "rb"))

# drop uneccessary columns
data_for_model.drop("customer_id", axis = 1, inplace = True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)
```
<br>
### Data Preprocessing <a name="regtree-preprocessing"></a>

While Linear Regression is susceptible to the effects of outliers, and highly correlated input variables - Decision Trees are not, so the required preprocessing here was lighter. I still however put in place logic for:

* Missing values in the data
* Encoding categorical variables to numeric form

<br>
##### Missing Values

The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) I just removed those rows.

```python
# remove rows where values were missing
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)
```

<br>
##### Split Out Data For Modelling

In exactly the same way I did for Linear Regression, I splitted my data into an **X** object which contained only the predictor variables, and a **y** object which contained only my dependent variable.

Once I did this, I splitted my data into training and test sets to ensure I could fairly validate the accuracy of the predictions on data that was not used in training. In this case, I allocated 80% of the data for training, and the remaining 20% for validation.

<br>
```python
# split data into X and y objects for modelling
X = data_for_model.drop(["customer_loyalty_score"], axis = 1)
y = data_for_model["customer_loyalty_score"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```

<br>
##### Categorical Predictor Variables

In my dataset, I had one categorical variable *gender* which had values of "M" for Male, "F" for Female, and "U" for Unknown.

As *gender* doesn't have any explicit *order* to it, in other words, Male isn't higher or lower than Female and vice versa - we would again apply One Hot Encoding to the categorical column.

```python
# list of categorical variables that need encoding
categorical_vars = ["gender"]

# instantiate OHE class
one_hot_encoder = OneHotEncoder(sparse=False, drop = "first")

# apply OHE
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# extract feature names for encoded columns
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# turn objects back to pandas dataframe
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)
```

<br>
### Model Training <a name="regtree-model-training"></a>

Instantiating and training the Decision Tree model was done using the below code. I used the *random_state* parameter to ensure I got reproducible results, and this helped us understand any improvements in performance with changes to model hyperparameters.

```python
# instantiate my model object
regressor = DecisionTreeRegressor(random_state = 42)

# fit the model using my training & test sets
regressor.fit(X_train, y_train)
```

<br>
### Model Performance Assessment <a name="regtree-model-assessment"></a>

##### Predict On The Test Set

To assess how well my model was predicting new data - I used the trained model object (here called *regressor*) and asked it to predict the *loyalty_score* variable for the test set.

```python
# predict on the test set
y_pred = regressor.predict(X_test)
```

<br>
##### Calculate R-Squared

```python
# calculate r-squared for our test set predictions
r_squared = r2_score(y_test, y_pred)
print(r_squared)
```

The resulting r-squared score from this was **0.898**.

<br>
##### Calculate Cross Validated R-Squared

As I did when testing Linear Regression, I utilized Cross-Validation.

Instead of simply dividing my data into a single training set, and a single test set, with Cross Validation I broke my data into several "chunks" and then iteratively trained the model on all but one of the "chunks", tested the model on the remaining "chunk" until each had a chance to be the test set.

The result of this was that I was provided a number of test set validation results - and I took the average of these to give a much more robust & reliable view of how my model performed on new, unseen data!

In the code below, I put this into place. I again specified 4 "chunks" and then I passed in my regressor object, training set, and test set. I also specified the metric I wanted to assess with, in this case, I sticked to r-squared.

Finally, I took a mean of all four test set results.

```python
# calculate the mean cross-validated r-squared for my test set predictions
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()
```

The mean cross-validated r-squared score from this was **0.871** which was slighter higher than what I saw for Linear Regression.

<br>
##### Calculate Adjusted R-Squared

```python
# calculate adjusted r-squared for the test set predictions
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)
```

The resulting *adjusted* r-squared score from this was **0.887** which as expected, was slightly lower than the score I got for r-squared on its own.

<br>
### Decision Tree Regularisation <a name="regtree-model-regularisation"></a>

Decision Trees can be prone to over-fitting, in other words, without any limits on their splitting, they will end up learning the training data perfectly. We would much prefer our model to have a more *generalized* set of rules, as this will be more robust & reliable when making predictions on *new* data.

One effective method of avoiding this over-fitting is to apply a *max depth* to the Decision Tree, meaning we only allow it to split the data a certain number of times before it is required to stop.

Unfortunately, we don't necessarily know the *best* number of splits to use for this - so below I looped over a variety of values and assessed which gave me the best predictive performance!

<br>
```python
# finding the best max_depth

# set up range for search, and empty list to append accuracy scores to
max_depth_list = list(range(1,9))
accuracy_scores = []

# loop through each possible depth, train and validate model, append test set accuracy
for depth in max_depth_list:
    
    regressor = DecisionTreeRegressor(max_depth = depth, random_state = 42)
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    accuracy = r2_score(y_test,y_pred)
    accuracy_scores.append(accuracy)
    
# store max accuracy, and optimal depth    
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]

# plot accuracy by max depth
plt.plot(max_depth_list,accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy by Max Depth \n Optimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy,4)})")
plt.xlabel("Max Depth of Decision Tree")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()
```
<br>
That code gave me the below plot - which visualized the results!

![alt text](/img/posts/regression-tree-max-depth-plot.png "Decision Tree Max Depth Plot")

<br>
In the plot, I saw that the *maximum* classification accuracy on the test set was found when applying a *max_depth* value of 7. However, I lost very little accuracy back to a value of 4, but this would result in a simpler model, that generalized even better on new data. I made the executive decision to re-train my Decision Tree with a maximum depth of 4!

<br>
### Visualise the Decision Tree <a name="regtree-visualise"></a>

To see the decisions that have been made in the (re-fitted) tree, I used the plot_tree functionality that I imported from scikit-learn. To do this, I used the below code:

```python
# re-fit my model using a max depth of 4
regressor = DecisionTreeRegressor(random_state = 42, max_depth = 4)
regressor.fit(X_train, y_train)

# plot the nodes of the decision tree
plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 16)
```
<br>
That code givave me the below plot:

![alt text](/img/posts/regression-tree-nodes-plot.png "Decision Tree Max Depth Plot")

<br>
This was a very powerful visual, and one that could be shown to stakeholders in the business to ensure they understand exactly what was driving the predictions.

One interesting thing to note was that the *very first split* appeared to be using the variable *distance from store* so it would seem that this was a very important variable when it came to predicting loyalty!

___
<br>
# Random Forest <a name="rf-title"></a>

I utilized the scikit-learn library within Python to model my data using a Random Forest. The code sections below were broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>
### Data Import <a name="rf-import"></a>

I imported my model, removed the id column, and shuffled the data.

```python
# import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

# import modeling data
data_for_model = pickle.load(open("data/customer_loyalty_modelling.p", "rb"))

# drop unnecessary columns
data_for_model.drop("customer_id", axis = 1, inplace = True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)
```
<br>
### Data Preprocessing <a name="rf-preprocessing"></a>

While Linear Regression is susceptible to the effects of outliers, and highly correlated input variables - Random Forests, just like Decision Trees, are not, so the required preprocessing here was lighter. I put in place logic for:

* Missing values in the data
* Encoding categorical variables to numeric form

<br>
##### Missing Values

The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) I just removed those rows.

```python
# remove rows where values were missing
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)
```

<br>
##### Split Out Data For Modelling

I first splitted the data into an **X** object which contained only the predictor variables, and a **y** object which contained only my dependent variable.

Once I did this, I splitted my data into training and test sets to ensure I could fairly validate the accuracy of the predictions on data that was not used in training. In this case, I allocated 80% of the data for training, and the remaining 20% for validation.

<br>
```python
# split data into X and y objects for modelling
X = data_for_model.drop(["customer_loyalty_score"], axis = 1)
y = data_for_model["customer_loyalty_score"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```

<br>
##### Categorical Predictor Variables

```python
# list of categorical variables that needed encoding
categorical_vars = ["gender"]

# instantiate OHE class
one_hot_encoder = OneHotEncoder(sparse=False, drop = "first")

# apply OHE
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# extract feature names for encoded columns
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# turn objects back to pandas dataframe
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)
```

<br>
### Model Training <a name="rf-model-training"></a>

Instantiating and training my Random Forest model was done using the below code. I used the *random_state* parameter to ensure I got reproducible results, and this helped me understand any improvements in performance with changes to model hyperparameters.

I left the other parameters at their default values, meaning that I just be building 100 Decision Trees in this Random Forest.

```python
# instantiate my model object
regressor = RandomForestRegressor(random_state = 42)

# fit my model using my training & test sets
regressor.fit(X_train, y_train)
```

<br>
### Model Performance Assessment <a name="rf-model-assessment"></a>

##### Predict On The Test Set

To assess how well my model was predicting new data - I used the trained model object (here called *regressor*) and asked it to predict the *loyalty_score* variable for the test set

```python
# predict on the test set
y_pred = regressor.predict(X_test)
```

<br>
##### Calculate R-Squared

```python
# calculate r-squared for our test set predictions
r_squared = r2_score(y_test, y_pred)
print(r_squared)
```

The resulting r-squared score from this was **0.957** - higher than both Linear Regression & the Decision Tree.

<br>
##### Calculate Cross Validated R-Squared

As I did when testing Linear Regression & our Decision Tree, I again utilized Cross Validation.
```python
# calculate the mean cross-validated r-squared for the test set predictions
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()
```

The mean cross-validated r-squared score from this was **0.923** which again was higher than I saw for both Linear Regression & our Decision Tree.

<br>
##### Calculate Adjusted R-Squared
```python
# calculate adjusted r-squared for our test set predictions
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)
```

The resulting *adjusted* r-squared score from this was **0.955** which as expected, was slightly lower than the score I got for r-squared on its own - but again higher than for my other models.

<br>
### Feature Importance <a name="rf-model-feature-importance"></a>

In my Linear Regression model, to understand the relationships between input variables and my output variable, loyalty score, I examined the coefficients. With my Decision Tree, I looked at what the earlier splits were. These allowed me some insight into which input variables were having the most impact.

Random Forests were an ensemble model, made up of many, many Decision Trees, each of which was different due to the randomness of the data being provided, and the random selection of input variables available at each potential split point.

Because of this, I ended up with a powerful and robust model, but because of the random or different nature of all these Decision trees - the model gave me a unique insight into how important each of my input variables was to the overall model.  

As I was using random samples of data, and input variables for each Decision Tree - there were many scenarios where certain input variables were being held back and this enabled me a way to compare how accurate the model's predictions were if that variable was or wasn’t present.

So, at a high level, in a Random Forest, I could measure *importance* by asking *How much would accuracy decrease if a specific input variable was removed or randomized?*

If this decrease in performance, or accuracy, was large, then I’d deem that input variable to be quite important, and if I saw only a small decrease in accuracy, then I’d conclude that the variable is of less importance.

At a high level, there were two common ways to tackle this. The first, often just called **Feature Importance** is where we find all nodes in the Decision Trees of the forest where a particular input variable is used to split the data and assess what the Mean Squared Error (for a Regression problem) was before the split was made, and compare this to the Mean Squared Error after the split was made.  We can take the *average* of these improvements across all Decision Trees in the Random Forest to get a score that tells us *how much better* we’re making the model by using that input variable.

If we do this for *each* of our input variables, we can compare these scores and understand which is adding the most value to the predictive power of the model!

The other approach, often called **Permutation Importance** cleverly uses some data that has gone *unused* at when random samples are selected for each Decision Tree (this stage is called "bootstrap sampling" or "bootstrapping")

These observations that were not randomly selected for each Decision Tree are known as *Out of Bag* observations and these can be used for testing the accuracy of each particular Decision Tree.

For each Decision Tree, all of the *Out of Bag* observations are gathered and then passed through. Once all of these observations have been run through the Decision Tree, we obtain an accuracy score for these predictions, which in the case of a regression problem could be Mean Squared Error or r-squared.

In order to understand the *importance*, we *randomize* the values within one of the input variables - a process that essentially destroys any relationship that might exist between that input variable and the output variable - and run that updated data through the Decision Tree again, obtaining a second accuracy score. The difference between the original accuracy and the new accuracy gives us a view on how important that particular variable is for predicting the output.

*Permutation Importance* is often preferred over *Feature Importance* which can at times inflate the importance of numerical features. Both are useful and in most cases will give fairly similar results.

Here, I put them both in place and plotted the results...

```python
# calculate feature importance
feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names,feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable","feature_importance"]
feature_importance_summary.sort_values(by = "feature_importance", inplace = True)

# plot feature importance
plt.barh(feature_importance_summary["input_variable"],feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# calculate permutation importance
result = permutation_importance(regressor, X_test, y_test, n_repeats = 10, random_state = 42)
permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names,permutation_importance], axis = 1)
permutation_importance_summary.columns = ["input_variable","permutation_importance"]
permutation_importance_summary.sort_values(by = "permutation_importance", inplace = True)

# plot permutation importance
plt.barh(permutation_importance_summary["input_variable"],permutation_importance_summary["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()
```
<br>
That code gave me the below plots - the first being for *Feature Importance* and the second for *Permutation Importance*!

![alt text](/img/posts/rf-regression-feature-importance.png "Random Forest Feature Importance Plot")

<br>
![alt text](/img/posts/rf-regression-permutation-importance.png "Random Forest Permutation Importance Plot")

<br>
The overall story from both approaches was very similar, in that by far, the most important or impactful input variable was *distance_from_store* which was the same insights I derived when assessing my Linear Regression & Decision Tree models.

There were slight differences in the order or "importance" of the remaining variables but overall they provided similar findings.

___
<br>
# Modelling Summary  <a name="modelling-summary"></a>

The most important outcome of this project was predictive accuracy, rather than explicitly understanding the drivers of prediction. Based upon this, I chose the model that performed the best when predicted on the test set - the Random Forest.

<br>
**Metric 1: Adjusted R-Squared (Test Set)**

* Random Forest = 0.955
* Decision Tree = 0.886
* Linear Regression = 0.754

<br>
**Metric 2: R-Squared (K-Fold Cross Validation, k = 4)**

* Random Forest = 0.925
* Decision Tree = 0.871
* Linear Regression = 0.853

<br>
Even though I was not specifically interested in the drivers of prediction, it was interesting to see across all three modelling approaches, that the input variable with the biggest impact on the prediction was *distance_from_store* rather than variables such as *total sales*. This was interesting information for the business, so discovering this as I went was worthwhile.

<br>
# Predicting Missing Loyalty Scores <a name="modelling-predictions"></a>

I selected Random Forest to make the *loyalty_score* predictions for those customers that the market research consultancy was unable to tag.

I could not just pass the data for these customers into the model, as was - I needed to ensure the data was in exactly the same format as what was used when training the model.

In the following code, I 

* Import the required packages for preprocessing
* Import the data for those customers who are missing a *loyalty_score* value
* Import our model object & any preprocessing artifacts
* Drop columns that were not used when training the model (customer_id)
* Drop rows with missing values
* Apply One Hot Encoding to the gender column (using transform)
* Make the predictions using .predict()

```python
# import required packages
import pandas as pd
import pickle

# import customers for scoring
to_be_scored = ...

# import model and model objects
regressor = ...
one_hot_encoder = ...

# drop unused columns
to_be_scored.drop(["customer_id"], axis = 1, inplace = True)

# drop missing values
to_be_scored.dropna(how = "any", inplace = True)

# apply one hot encoding (transform only)
categorical_vars = ["gender"]
encoder_vars_array = one_hot_encoder.transform(to_be_scored[categorical_vars])
encoder_feature_names = one_hot_encoder.get_feature_names(categorical_vars)
encoder_vars_df = pd.DataFrame(encoder_vars_array, columns = encoder_feature_names)
to_be_scored = pd.concat([to_be_scored.reset_index(drop=True), encoder_vars_df.reset_index(drop=True)], axis = 1)
to_be_scored.drop(categorical_vars, axis = 1, inplace = True)

# make our predictions!
loyalty_predictions = regressor.predict(to_be_scored)
```
<br>
Just like that, I made my *loyalty_score* predictions for these missing customers. Due to the impressive metrics on the test set, I was reasonably confident with these scores. This extra customer information will ensure my client can undertake more accurate and relevant customer tracking, targeting, and comms.

___
<br>
# Growth & Next Steps <a name="growth-next-steps"></a>


I could even look to tune the hyperparameters of the Random Forest, notably regularisation parameters such as tree depth, as well as potentially training on a higher number of Decision Trees in the Random Forest.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken to ensure that I have as much useful information available for predicting customer loyalty.
