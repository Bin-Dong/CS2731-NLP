# Load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import shared_methods
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import numpy as np

learning_rate = np.array([1, 0.1, 0.5, 0.25, 0.01, 0.05])
n_estimators = np.array([10,100,500,1000,1500])
hyperparameters = dict(learning_rate=learning_rate, n_estimators=n_estimators)

# Load data
text_list, label_list = shared_methods.read_csv("yelp_reviews_preprocessed.csv")
text_list, label_list = shuffle(text_list, label_list, random_state=1)

text_list_reduce = text_list[:1000]
label_list_reduce = label_list[:1000]


TFIDF_list, vectorizer = shared_methods.TFIDF_Vectorization(text_list_reduce)
# X_sparse = coo_matrix(TFIDF_list)


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(TFIDF_list, label_list_reduce, test_size=0.70) # 95% training and 30% test

# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=1000, learning_rate=0.01)
# abc = AdaBoostClassifier()


# clf = GridSearchCV(estimator=abc, param_grid=hyperparameters, cv=2, verbose=2, n_jobs=-1)

# best_model = clf.fit(X_train, y_train)

# print('Best Learning Rate:', best_model.best_estimator_.get_params())

# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
# text_list_test = vectorizer.transform(text_list_test).toarray().tolist()
# print(len(text_list_test))


y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

