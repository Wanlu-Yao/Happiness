# -*- coding: utf-8 -*-
"""
<Group_1_Happiness Survey In the City of Somerville>

Copyright (c) 2020
Licensed
Written by <Yinuo Pan and Wanlu Yao>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.formula.api as smf
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore") 
############################################################## Section One #########################################################################
#Get the original dataframe
def GET_DATA(url):
    df = pd.read_csv(url,header = 0)
    '''
    ：param url: the url of data set
    ：variable df:Read data from url,save as dataframe
    ：return: return the dataframe df
   
    '''
    return df

#Modify the dataframe used in Section 1
def GET_DATA_1(df):
    new_df = df.iloc[:,4:17]
    new_df.columns = ['Overall satisfaction',
                      'Score of Neighborhood quality',
                      'Score of Community honor',
                      'Score of City services',
                      'Score of Cost of housing',
                      'Score of Quality of public schools',
                      'Score of Local police','Score of Streets and sidewalks',
                      'Score of Social community events',
                      'Score of Transportation options',
                      'Score of Safe at night',
                      'Score of Parks and squares',
                      'Score of Physical setting of neighborhood']
    '''
    ：param df: the original dataframe
    ：variable new_df: Get the required columns as a subset and rename them
    : return: the subset new_df
    '''
    return new_df

#Clean up the dataframe used for Section 1    
def CLEAN_DATA_1(df):
    df = df.dropna(how = 'any') 
    '''
    ：param df: the original dataframe
    Delete all NA values in the dataframe df
    : return: cleaned df
    '''
    return df

#Draw a combination diagram of pairwise relationship
def DRAW_PAIR_PLOT(x,y,df):
    p = sb.jointplot(x = x, y = y, data = df,kind = 'kde')
    p.fig.suptitle(f'Kernel density estimate of {x} and {y}')
    '''
    ：param x: the value of x axis
    ：param y: the value of y axis
    ：param df: the data we use for create the plot
    ：variable p: use seaborn to draw pair plots
    Edit the title of the plot according to the drawn variables
    '''
    

#Plot the frequency distribution
def DRAW_DIS_PLOT(df,cols_name):
    '''
    ：param df: the dataframe we use for creating the plot
    ：param cols_name: the column we need to analysis
    '''
    sb.distplot(df[cols_name])
    
    
def GET_DF_3(df):
    df3 = df.iloc[:,4:17]
    df3.columns = ['Overall_satisfaction',
                      'Neighborhood',
                      'Pride',
                      'city services',
                      'housing',
                      'public_schools',
                      'local police','Streets and sidewalks',
                      'social community events',
                      'Transportation options',
                      'Safe at night',
                      'Parks_and_squares',
                      'Physical_setting']
    '''
    ：param df: the original dataframe
    Slice subsets and rename columns
    :return: the subsets df3
    '''
    return df3

def GET_RESULT(df3):
    results_1 = smf.ols('Overall_satisfaction ~ Neighborhood+ Pride + Parks_and_squares', data = df3).fit()
    print(results_1.params)
    results_1.summary()
    '''
    ：param df3: the dataframe we used for modeling
    ：variable: results_1: fit the linear model based on the top three factors of impression
    Get the report of the model
    :return: the model results_1
    '''
    return results_1
    
    
############################################################### Section Two ########################################################################

def GET_DATA_2(url):
    df = pd.read_csv(url, dtype={'2. Age?': object})
    df.head()
    df.describe(include='all')
    df.isna().sum()
    '''
    ：param url: the url we get the data set
    ：variable df: Get data from url and encode according to the required format
    Check the describe of the data and the NA value
    :return: the dataframe df for further use
    '''
    return df

def DATA_CLEAN_2(df):
    df1= df.iloc[:,17:27]
    df1.columns =['gender','age','language','race','children age 18 or younger','housing status','move away plan','duration of residence','annual income','student or not']
    df1['satisfaction'] = df['4. How satisfied are you with Somerville as a place to live?']
    df1 = df1[df1['satisfaction'].notna()]
    df1['gender'].fillna("Unknown", inplace = True)
    df1['age'].fillna("Unknown", inplace = True)
    df1['race'].fillna("Unknown", inplace = True)
    df1['children age 18 or younger'].fillna("Unknown", inplace = True)
    df1['housing status'].fillna("Unknown", inplace = True)
    df1['duration of residence'].fillna("Unknown", inplace = True)
    df1['annual income'].fillna("Unknown", inplace = True)
    df1['high satisfaction']= np.where(df1.satisfaction>= 8,True, False)
    '''
    ：param df: the original dataframe
    ：variable df1: Take a subset and rename
    Ensure that the satisfaction(Representative total score) is not NA
    Classify other missing values as unknown
    ：variable name 'high satisfication': When satisfaction is greater than or equal to 8, it means high satisfaction is TRUE
    ：return: the new subset df1
    '''
    return df1

def DATA_NEW_2(df1):
    df2 = df1[df1['high satisfaction'] == True]
    '''
    ：param df1: the original 
    ：variable df2: Get a subset of TRUE with high satisfaction
    ：return: the subset df2 with high satisfaction
    '''
    return df2

def S2_PIE_CHART(df2,a,b,title):
    df2.loc[df2[a].isin(b)== False, a]='other'
    fig1= df2.groupby([a]).sum().plot(kind='pie', y='high satisfaction', autopct='%1.0f%%')
    fig1.set_title(title)
    plt.show()
    '''
    ：param df2: the dataframe we used for creating the chart
    ：param a: the column we use for create the chart
    ：param b: the remain groups we want to show
    ：param title: the title of the plot
    Classify minorities as "other"
    Draw a pie chart and fill in the title,Demonstrate the characteristics of highly satisfied people
    
    '''

def S2_PIE_CHART2(df2):
    fig2= df2.groupby(['children age 18 or younger']).sum().plot(kind='pie', y='high satisfaction', autopct='%1.0f%%')
    fig2.set_title("High Satisfaction Between Children ")
    plt.show()
    '''
    ：param df2: the dataframe we used for creating the chart
    For the columns with only TRUE or FALSE or UNKNOWN, we directly aggregate them
    Draw a pie chart and fill in the title,Demonstrate the characteristics of highly satisfied people

    '''


def AGE(df2):
    #x= df2.groupby('age')
    #y= df2['high satisfication'].value_counts() 
    plt.xlabel("Age Group")
    plt.ylabel("High Satisfaction Count")
    plt.title('Satisfaction Level Between Age Group')
    plt.xticks(rotation=15)
    df2.groupby('age').age.hist()
    '''
    ：param df2: the dataframe we used for creating the plot
    Set axis label and title
    Adjust for easy viewing
    Classification and summary according to various age groups
    
    '''

def BAR(data,x,pal,order,title,r):
    ax=sb.countplot(x=x, 
                   data= data, 
                   palette=pal,
                   order = order)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=r, ha="right")
    ax.set(title= title)
    fig_dims = (6, 4)
    '''
    ：param data: the data used for analysis
    ：param x: the groups we classification and summary
    ：param pal: set yhe palette parameter
    ：param order: The sorting of each group on the x-axis is convenient for sorting from small to large/from high to low
    ：param title: title of the plot
    ：param r:set the rotation parameter
    '''

def RACE_DF(df2):
    df4 = df2.copy()
    frequencies = df4["race"].value_counts(normalize = True)
    frequencies
    threshold = 0.02
    small_categories = frequencies[frequencies < threshold].index
    small_categories
    df4["race"] = df4["race"].replace(small_categories, "Other")
    df4["race"].value_counts(normalize = True)
    '''
    ：param df2: the original dataframe
    ：variable df4: copy of the df2
    ：variable frequencies: View the frequency of occurrence of each group in the race column
    ：variable threshold: Set a threshold to find the minority
    ：variable small_categories: Races with frequency below the threshold
    ：return: the new dataframe df4,which replaced the minority to "other"
    '''
    return df4



############################################################## Section Three#####################################################################

def GET_DATA_TREE(df1):
    DataTree = df1.drop(['satisfaction','language','move away plan'],axis = 1)
    DataTree.columns = ['gender','age','race','children','housing_status','duration','annual_income','student','High_Satisfaction']
    DataTree = RACE_DF(DataTree)
    DataTree.loc[DataTree['gender'].isin(['Female','Male'])== False, 'gender']='other'
    DataTree.dropna(inplace=True)
    DataTree = DataTree[~DataTree.eq('Unknown').any(1)]
    #print(DataTree.head())

    clos = ['gender','age','race','children','housing_status','duration','annual_income','student','High_Satisfaction']

    for clo in clos:
        u = DataTree[clo].unique()
        print(u)

        def conver(x):
            return np.argwhere(u==x)[0,0]

        DataTree[clo] = DataTree[clo].map(conver)
    import warnings
    warnings.filterwarnings("ignore") 

    '''
    ：param df1: the original dataframe
    ：variable DataTree:The data obtained for tree modeling has been preliminarily sorted out, but it is still with text structure
    ：variable clos: The name of the column that needs to be encoded
    ：variable u: list of groups in column, unique value,the index is their encoding
    ：variable x: The value in the column is used in the function to compare the group to which they belong, and grant the corresponding encoding
    ：local function conver: encode the value in the column
    ：return: the dataframe DataTree which is prepared for tree modeling
    '''
    return DataTree

def GET_DATA_FOR_TREE(DataTree,splitrate):
    X = DataTree[['gender','age','race','children','housing_status','duration','annual_income','student']]
    Y = DataTree['High_Satisfaction']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=splitrate, random_state=123)


    vec = DictVectorizer(sparse=False)
    #print(X_train.to_dict(orient='record'))
    X_train = vec.fit_transform(X_train.to_dict(orient='record'))
    #print(X_train)
    #print(vec.feature_names_)
    X_test = vec.transform(X_test.to_dict(orient='record'))
    import warnings
    warnings.filterwarnings("ignore") 
    
    '''
    ：param DataTree: the dataframe we use for create the vector data
    ：param splitrate: Divide the proportion of train set and test set
    ：variable X: Subset of factors
    ：variable Y: the target
    ：variable vec:Vector function, the tree model needs a format like file.data and file.target
    ：variable 
    X_train and X_test: According to the format required by the tree model, convert the value of X into a dictionary format and pack it into a vector
   
    ：return: the data(Xs and Y) for train and test
    '''
    
    return X_train, X_test, Y_train, Y_test

def BULID_DT(X_train, X_test, Y_train, Y_test):
    #Decision tree
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, Y_train)
    dtc_y_predict = dtc.predict(X_test)


    #Random forest
    rfc = RandomForestClassifier()
    rfc.fit(X_train, Y_train)
    rfc_y_predict = rfc.predict(X_test)



    #Gradient Boosting Decision Tree
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, Y_train)
    gbc_y_predict = gbc.predict(X_test)
    
    '''
    ：param X_train,Y_train and X_test,Y_test: the train and test data
    ：variable dtc: Fit the decision tree model
    ：variable rfc: Fit the random forest model
    ：variable gbc: Fit the Gradient Boosting Decision Tree model
    ：variable dtc_y_predict: the result of predicted y by using decision tree model
    ：variable rfc_y_predict: the result of predicted y by using random forest model
    ：variable gbc_y_predict: the result of predicted y by using Gradient Boosting Decision Tree model
    
    ：return: models and the predicted ys
    '''
    
    return dtc,rfc,gbc,dtc_y_predict,rfc_y_predict,gbc_y_predict

def REPORT_TREE(dtc,rfc,gbc,X_test, Y_test,dtc_y_predict,rfc_y_predict,gbc_y_predict):
    print(dtc.score(X_test, Y_test))
    print(classification_report(dtc_y_predict, Y_test))
    print(rfc.score(X_test, Y_test))
    print(classification_report(rfc_y_predict, Y_test))
    print(gbc.score(X_test, Y_test))
    print(classification_report(gbc_y_predict, Y_test))
    
    '''
    ：param dtc,rfc,gbc: models we built
    ：param X_test, Y_test: the test data set
    ：param dtc_y_predict,rfc_y_predict,gbc_y_predict: the predicted ys of different model
    Output the accuracy of each model and the overall report
    '''
