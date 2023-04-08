#                                           Democratic Sun Rises 1.0


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

class Regressor:
    def __init__(self, data):
        self.data = data
    
    def preprocess_data(self):
        # Convert text col to numerical values
        self.data = pd.get_dummies(self.data, columns=['Subject'])
        
        # Convert for/against to numerical values **arguments to change depending upong roll call data yay's & nay's
        self.data['Outcome'] = self.data['Outcome'].map({'For': 1, 'Against': 0})
        
    def train_models(self):
        # Cleanse
        X = self.data.drop('Outcome', axis =1)
        y = self.data['Outcome']
        
        #Train linear regression model
        self.linreg = LinearRegression()
        self.linreg.fit(X, y)
        
        #Train decision tree model
        self.tree = DecisionTreeRegressor()
        self.tree.fit(X, y)
        
        #Train random forest model
        self.randomForests = RandomForestRegressor()
        self.randomForests.fit(X, y)
        
    def predict(self,subject):
        subject_df = pd.DataFrame([[subject]], columns=['Subject'])
        subject_encoded = pd.get_dummies(subject_df)['Subject_'+ subject]
        
        #Combine feature vector with all other features
        features = pd.concat([pd.DataFrame([0], columns=['Outcome']), subject_encoded], axis=1)
        
        #Predict
        linreg_predict = self.linreg.predict(features)[0]
        tree_predict = self.tree.predict(features)[0]
        randomForest_predict = self.randomForests.predict(features)[0]
        
        # Results
        print('Linear Regression predicts -> ', linreg_predict)
        print('Decision Tree predicts -> ', tree_predict)
        print('Random Forests predicts -> ', randomForest_predict)
        
        #which model is most accurate?
        if linreg_predict > tree_predict and linreg_predict > randomForest_predict:
            print('Linear regression is the most accurate model.')
        elif tree_predict > linreg_predict and tree_predict > randomForest_predict:
            print('Decision tree is the most accurate model.')
        else:
            print('Random forest is the most accurate model')

data = pd.DataFrame([['Math', 'Yes'],
                     ['Science', 'No'],
                     ['History', 'No'],
                     ['English', 'Yes']], columns=['Subject', 'Outcome'])

regressor = Regressor(data)    #enter in frame label here
regressor.preprocess_data()
regressor.train_models()

regressor.predict('Math')     #enter in term for prediction of yay's & nay's 