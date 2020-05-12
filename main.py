import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import NeuralNet as nn
from sklearn.model_selection import train_test_split

# Data loading
gaussian_df = pd.read_csv( 'gaussian_data.csv' )
# Setting aside a portion of the data for the final test of our model
test_final_df =  gaussian_df.sample(frac=0.2, random_state=42)
# Removing that part from data
gaussian_df = gaussian_df.drop( test_final_df.index )
# Getting the features for train and final test
X = gaussian_df.drop( "class", axis = 1 )
X_test_final = test_final_df.drop( "class", axis = 1 )
# Getting the labels for train and final test
y = gaussian_df["class"]
y = pd.get_dummies(y)
# To one-hot
y_test_final = test_final_df["class"]
y_test_final = pd.get_dummies(y_test_final)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# Initialization of the model
model = nn.NeuralNet(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)
# Fitting the model with data
gr_train, gr_test = model.train( X_train, y_train, X_test, y_test )
# Prediction for being sure that everything is OK 
acc,prr = model.predict(X_test_final,y_test_final)
# Confusion matrix
conf_mx = model.confusion_matrix(prr,y_test_final)
print()
#Plotting the graph of the errors 
plt.plot(gr_train, label='train')
plt.plot(gr_test, label='test')
plt.legend()
plt.show()
print("Last error: "+ str(gr_train[-1]))
