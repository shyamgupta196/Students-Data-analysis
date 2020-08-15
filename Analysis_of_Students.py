# Problems
'''

1.how do differnt times affects grades of students

2.how to multiple factors affected grades of students

3.what is the relation between each category 
# motive 
to get thorogh idea of about how to make multiple regression on datasets
And  make my self comfortable in using regression models also saving models with the help of pickle for further predictions..


'''

# performing EDA on DataSet

'''for EDA'''
import pandas as pd 
import numpy as np 
from scipy import stats as st
'''for visualisation purposes'''
import matplotlib.pyplot as plt
import seaborn as sns



pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

new_df=pd.read_csv('student-mat.csv',sep=';')
obj_copy=new_df.select_dtypes(include='object').copy()

print(obj_copy.head())
manipulated_df=pd.get_dummies(obj_copy,columns=['sex','Pstatus','activities','internet','romantic'],drop_first=True)
# pd.set_option()
print(manipulated_df.head())
num_df=new_df.select_dtypes(include=np.number).copy()

DATA=pd.concat([num_df.loc[:,'age':'absences'],manipulated_df.loc[:,'sex_M':]],axis=1)

Grades=num_df[['G1','G2','G3']]

DATA=pd.concat([DATA,Grades],axis=1)
# We see Many Features are not of any USE for EDA so we Drop them
DATA.drop(columns=['famrel','failures','Medu','Fedu'],inplace=True)
#we see how many people are there of diff ages
#handling of outliers
print(DATA.age.value_counts())
DATA.drop(DATA.loc[(DATA['age']>=20)].index,inplace=True)

'''visualisation'''

def plotter(df,compare_with,typo_of_graph):
   '''
   takes in all column names and creates a graph between their relation with compare with feature other 
   '''
   for feature in df.columns:
       if not feature==compare_with:
           df.plot(compare_with,feature,kind=typo_of_graph)
           
       else:
           pass 

##plotter(DATA,'G3','kde')

'''
people with average no of outings do have a high chance of scoring good marks telling that its healthy and effects positively on brain
than people with no kind of outings 
'''

#hence we can drop free time and age also

# DATA.drop(axis=1,columns=["sex_M"
#        ],inplace=True)

##DATA.plot(x='G3',y='goout',kind='hexbin')
##plt.show()
sns.kdeplot(DATA['G3'],DATA['goout'],cmap='PuBu',n_levels=20,shade=True)
plt.show()
print(DATA)
print(DATA.columns)

'''
From the data its clear that
#romantic
students in relation have lesser marks as compared to those who were not
#activities
students engaged in activities were were able to score more
reflecting the role of sports
#parents_together
students whose parents stayed together had higher chances
of scoring good
#sex_chances
females scored more as compared to men
#health
students having good health max of them scored good 
#AGE(not a gr8 factor)
Age didnt became a great factor in affecting the scores with the available data
#Walc & Dalc
#alcohol is not good for the body of students hence
students who consumed very low alcohol weekly scored more
students who consumed very low alcohol daily basis scored more


#travel_time seriously had an impact on the students performance

hence students who took less than 15 mins to reach had a very good chance
in scoring good marks and also who took more than at hour had a very low chance
of scoring (the analysis is limited to the data only

#students studying 2-5 hrs a week have a gr8 ability to score
'''
'''
since now the data is cleaned we can save it to a csv file

'''
DATA.to_csv('cleaned-Students-DATA.csv')

'''
creating a model
simple linear and multiple regression
'''
#Reading DATA

clean_data=pd.read_csv('cleaned-Students-DATA.csv')

###we plot all the scores on a kde plot to see what are the range of values getting predicted accurately

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


scores=[]
def score_plot():
    for i in range(1,40):
        
        
        x=np.array(clean_data.drop(["G3"],1))
        y=np.array(clean_data['G3'])
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)

        predictor=LinearRegression()

        predictor.fit(x_train,y_train)
        ##    x.values.reshape(-1,1)
        ##    
        ##    pred_model=predictor.predict(x_test)
        #     predictor.predict([[16,1,3,1,3,1,1,3,9,1,0,1,0,1,14]])
        #     predictor.score(x_test,y_test)
        scores.append(predictor.score(x_test,y_test))
    return


score_plot()

sns.kdeplot(scores)
plt.hist(scores,bins=[0.70,0.80,0.90,1])
plt.show()

# the graphs are saved in the repository for viewing purposes
'''

got a good accuracy of range(75-95)% 
could have been better if decision tree or advanced algorithm 
applied on to the analysis

''' 


