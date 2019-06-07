# https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
#%% loading data
from sklearn import datasets
wine = datasets.load_wine()
#%% exploring data
print ("feature names: ", wine.feature_names)  # pylint: disable=maybe-no-member
print ("label names: ", wine.target_names)  # pylint: disable=maybe-no-member
print ("shape: ", wine.data.shape) # pylint: disable=maybe-no-member
print ("top 5 records of features: ", wine.data[0:5]) # pylint: disable=maybe-no-member
print ("labels: ", wine.target) # pylint: disable=maybe-no-member
#%% splitting data
from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(
    wine.data, wine.target, test_size=0.3, random_state=109  # pylint: disable=maybe-no-member
)

#%% Training and prediction
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(feature_train, label_train)
pred = gnb.predict(feature_test)
#%% evaluating model
from sklearn import metrics
print ("Accuracy:", metrics.accuracy_score(label_test, pred))

