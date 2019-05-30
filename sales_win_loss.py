#%% link: https://www.dataquest.io/blog/sci-kit-learn-tutorial/
import pandas as pd
url = "https://community.watsonanalytics.com/wp-content/uploads/2015/04/WA_Fn-UseC_-Sales-Win-Loss.csv"
sales_data = pd.read_csv(url)

#%% Explore Data 
#%%
sales_data.head(n=2)
#%%
sales_data.tail(n=2)
#%%
sales_data.dtypes

#%% Visualize Data 
import seaborn as sns 
import matplotlib.pyplot as plt
#background color
sns.set(style="whitegrid", color_codes=True)
#plot size
sns.set(rc={'figure.figsize':(11.7,8.27)})
# x-axis is set to 'Route to Market'

sns.countplot('Route To Market',data=sales_data,hue = 'Opportunity Result')
sns.despine(offset=10, trim=True)
plt.show()

#%% violinplot
#  a violin plot displays the distribution of data across labels
import seaborn as sns 
import matplotlib.pyplot as plt 
sns.set(rc={'figure.figsize':(16.7,13.27)})
sns.violinplot(x="Opportunity Result", y="Client Size By Revenue", hue ="Opportunity Result", data=sales_data)
plt.show()
#%% Categorical columns
print("Supplies Subgroup : ", sales_data['Supplies Subgroup'].unique())
print("Region : ", sales_data['Region'].unique())
print("Route To Market : ", sales_data['Route To Market'].unique())
print("Opportunity Result : ", sales_data['Opportunity Result'].unique())
print("Competitor Type : ", sales_data['Competitor Type'].unique())
print("Supplies Group : ", sales_data['Supplies Group'].unique())

#%% Encode strings into numeric labels 
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
sales_data['Supplies Subgroup'] = le.fit_transform(sales_data['Supplies Subgroup'])
sales_data['Region'] = le.fit_transform(sales_data['Region'])
sales_data['Route To Market'] = le.fit_transform(sales_data['Route To Market'])
sales_data['Opportunity Result'] = le.fit_transform(sales_data['Opportunity Result'])
sales_data['Competitor Type'] = le.fit_transform(sales_data['Competitor Type'])
sales_data['Supplies Group'] = le.fit_transform(sales_data['Supplies Group'])
sales_data.head()
#%% Training Set & Test Set
# Need to run the above cell to pre-process the data
cols =[col for col in sales_data.columns if col not in ['Opportunity Number', 'Opportunity Result']]
data = sales_data[cols]
target = sales_data['Opportunity Result']
#%%
target.head(n=2)
#%%
# feature set
data.head(n=2)
#%% 
# Split the data into training set and test set
from sklearn.model_selection import train_test_split
# Set the random_state to a certain number, doesn’t matter what it actually is, will guarantee how the data is split will remain the same every time you run this method. 
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size= 0.30, random_state = 10) 

#%%
data_train.head(n=2)

#%% Gaussian Naïve Bayes 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
gnb = GaussianNB()
# train the algorithm
pred = gnb.fit(data_train, target_train).predict(data_test)
print (pred.tolist())
#%% GNB accuracy score
print("Naive-Bayes accuracy : ", accuracy_score(target_test, pred, normalize= True))
#%% GNB Performance
from yellowbrick.classifier import ClassificationReport
visualizer = ClassificationReport(gnb, classes = ['Won', 'Loss'])
visualizer.fit(data_train, target_train)
visualizer.score(data_test, target_test)
g = visualizer.poof()

#%% Linear Support Vector Classification
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
svc = LinearSVC(random_state=0)
# Need to investigate this warning: Liblinear failed to converge, increase the number of iterations.
# "the number of iterations.", ConvergenceWarning)

pred = svc.fit(data_train, target_train).predict(data_test)
print ("LinearSVC accuracy : ", accuracy_score(target_test, pred, normalize=True))

#%% SVC Performance
from yellowbrick.classifier import ClassificationReport
visualizer = ClassificationReport(svc, classes = ['Won', 'Loss'])
visualizer.fit(data_train, target_train)
visualizer.score(data_test, target_test)
g = visualizer.poof()

#%% K-Neighbors Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knc = KNeighborsClassifier(n_neighbors=3)
knc.fit(data_train, target_train)
pred = knc.predict(data_test)
print ("K-Neighbors accuracy score : ", accuracy_score(target_test, pred))

#%% K-Neighbors Classifiers performance
from yellowbrick.classifier import ClassificationReport
visualizer = ClassificationReport(knc, classes = ['Won', 'Loss'])
visualizer.fit(data_train, target_train)
visualizer.score(data_test, target_test)
g = visualizer.poof()
