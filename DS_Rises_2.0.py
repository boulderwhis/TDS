import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

class SubjectImpactPredictor:
    
    def __init__(self, df):
        self.df = df
        self.X = df.iloc[:, :-1] # select all columns except the last one
        self.y = df.iloc[:, -1] # select the last column
        
    def select_method(self):
        """
        Determine which machine learning method is most appropriate based on the type of data in the dataframe
        """
        # check if the target variable is binary (yes or no)
        if set(self.y) == {'yes', 'no'}:
            # use logistic regression for binary classification
            return 'Logistic Regression'
        
        # check if the target variable is categorical
        elif self.y.dtype == 'object':
            # use decision tree classifier for categorical classification
            return 'Decision Tree Classifier'
        
        # check if the target variable is continuous
        elif self.y.dtype == 'float':
            # use linear regression for continuous prediction
            return 'Linear Regression'
        
        else:
            return 'Target variable type not recognized'
    
    def predict_impact(self):
        """
        Predict how the subjects impact the target variable
        """
        # determine which machine learning method to use
        method = self.select_method()
        
        # split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        # use the appropriate machine learning method
        if method == 'Logistic Regression':
            model = LogisticRegression(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return y_pred
        
        elif method == 'Decision Tree Classifier':
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return y_pred
        
        elif method == 'Linear Regression':
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return y_pred
        
        else:
            return 'Error: Machine learning method not recognized'



# create a dataframe
df = pd.DataFrame({'Subject1': [1, 0, 1, 0, 1], 'Subject2': [0, 1, 0, 1, 1], 'Target': ['yes', 'no', 'yes', 'no', 'yes']})

# create an instance of the SubjectImpactPredictor class
impact_predictor = SubjectImpactPredictor(df)

# call the predict_impact method to predict how the subjects impact the target variable
y_pred = impact_predictor.predict_impact()
sub_pred = impact_predictor.select_method()
print(y_pred)
print(sub_pred)
