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
sales_data['Supplies Group'] = le.fit_transform(sales_data['Supplies Group'])
sales_data.head() 