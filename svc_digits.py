#%%
from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits() 
#%%
#(n_samples, n_features) (1797, 64)
# 8x8 image so the number of features per sample is 8x8 = 64 
print (digits.data) # features # pylint: disable=maybe-no-member

print (digits.data.shape) # pylint: disable=maybe-no-member

print (digits.target) # pylint: disable=maybe-no-member
print (digits.target.shape) # pylint: disable=maybe-no-member

print (digits.images[0]) # pylint: disable=maybe-no-member

#%%
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])  # pylint: disable=maybe-no-member  
clf.predict(digits.data[-1:]) # pylint: disable=maybe-no-member