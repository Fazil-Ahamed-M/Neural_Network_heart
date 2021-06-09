#prepare data downloaded from UCL

import pandas as pd
import numpy as np 
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore") #supress warnings
pd.set_option("display.max_rows", None, "display.max_columns", None)

# add header names
headers =  ['age', 'sex','chest_pain','resting_blood_pressure',  
        'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
        'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',"slope of the peak",
        'num_of_major_vessels','thal', 'heart_disease']

heart_df = pd.read_csv('heart.dat', sep=' ', names=headers)

#------Exploring the data------
# print(heart_df.head())   
# print(heart_df.shape)
# print(heart_df.isna().sum())
# print(heart_df.dtypes)

x = heart_df.drop(columns=['heart_disease'])

heart_df['heart_disease'] = heart_df['heart_disease'].replace(1,0)
heart_df['heart_disease'] = heart_df['heart_disease'].replace(2,1)

y = heart_df['heart_disease'].values.reshape(x.shape[0], 1)

#split data into train and test sets
x_train, x_test = train_test_split(x, test_size=0.2, random_state=2, shuffle=False)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=2, shuffle=False)


#standardize the dataset
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

print(f"Shape of train set is {x_train.shape}")
print(f"Shape of test set is {x_test.shape}")
print(f"Shape of train label is {y_train.shape}")
print(f"Shape of test labels is {y_test.shape}")

class NeuralNet():
    '''
    A two layer neural network
    '''
        
    def __init__(self, layers=[13,8,1], learning_rate=0.001, iterations=100):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.x = None
        self.y = None
                
    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution
        '''
        np.random.seed(1) # Seed the random number generator
        self.params["w1"] = np.random.randn(self.layers[0], self.layers[1]) 
        self.params['b1']  =np.random.randn(self.layers[1])
        self.params['w2'] = np.random.randn(self.layers[1],self.layers[2]) 
        self.params['b2'] = np.random.randn(self.layers[2])
    
    def relu(self,z):
        return np.maximum(0,z)

    def dRelu(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def eta(self, x):
      ETA = 0.0000000001
      return np.maximum(x, ETA)


    def sigmoid(self,z):
        '''
        The sigmoid function takes in real numbers in any range and 
        squashes it to a real-valued output between 0 and 1.
        '''
        return 1/(1+np.exp(-z))

    def entropy_loss(self,y, yhat):
        nsample = len(y)
        yhat_inv = 1.0 - yhat
        y_inv = 1.0 - y
        yhat = self.eta(yhat) ## clips value to avoid NaNs in log
        yhat_inv = self.eta(yhat_inv) 
        loss = -1/nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((y_inv), np.log(yhat_inv))))
        return loss

    def forward_propagation(self):
        '''
        Performs the forward propagation
        '''
        
        z1 = self.x.dot(self.params['w1']) + self.params['b1']
        a1 = self.relu(z1)
        z2 = a1.dot(self.params['w2']) + self.params['b2']
        yhat = self.sigmoid(z2)
        loss = self.entropy_loss(self.y,yhat)

        # save calculated parameters     
        self.params['z1'] = z1
        self.params['z2'] = z2
        self.params['a1'] = a1
        return yhat,loss


    def back_propagation(self,yhat):
        '''
        Computes the derivatives and update weights and bias according.
        '''
        y_inv = 1 - self.y
        yhat_inv = 1 - yhat

        loss_wrt_yhat = np.divide(y_inv, self.eta(yhat_inv)) - np.divide(self.y, self.eta(yhat))
        loss_wrt_sig = yhat * (yhat_inv)
        loss_wrt_z2 = loss_wrt_yhat * loss_wrt_sig

        loss_wrt_a1 = loss_wrt_z2.dot(self.params['w2'].T)
        loss_wrt_w2 = self.params['a1'].T.dot(loss_wrt_z2)
        loss_wrt_b2 = np.sum(loss_wrt_z2, axis=0, keepdims=True)

        loss_wrt_z1 = loss_wrt_a1 * self.dRelu(self.params['z1'])
        loss_wrt_w1 = self.x.T.dot(loss_wrt_z1)
        loss_wrt_b1 = np.sum(loss_wrt_z1, axis=0, keepdims=True)

        #update the weights and bias
        self.params['w1'] = self.params['w1'] - self.learning_rate * loss_wrt_w1
        self.params['w2'] = self.params['w2'] - self.learning_rate * loss_wrt_w2
        self.params['b1'] = self.params['b1'] - self.learning_rate * loss_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * loss_wrt_b2


    def fit(self, x, y):
        '''
        Trains the neural network using the specified data and labels
        '''
        self.x = x
        self.y = y
        self.init_weights() #initialize weights and bias

        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)

        
    def predict(self, x):
        '''
        Predicts on a test data
        '''
        z1 = x.dot(self.params['w1']) + self.params['b1']
        a1 = self.relu(z1)
        z2 = a1.dot(self.params['w2']) + self.params['b2']
        pred = self.sigmoid(z2)
        return np.round(pred) 

    def acc(self, y, yhat):
        '''
        Calculates the accutacy between the predicted valuea and the truth labels
        '''
        acc = int(sum(y == yhat) / len(y) * 100)
        return acc

    def plot_loss(self):
        '''
        Plots the loss curve
        '''
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("logloss")
        plt.title("Loss curve for training")
        plt.show()


'''
Initializing the nerual network - 1
'''
nn = NeuralNet()            # Create the nn model 
nn.fit(x_train, y_train)    # Train the model

nn.plot_loss()
train_pred = nn.predict(x_train)
test_pred = nn.predict(x_test)

print("Train accuracy of first network is {}".format(nn.acc(y_train, train_pred)))
print("Test accuracy of first network is {}".format(nn.acc(y_test, test_pred)))



'''
Initialzing the neural network - 2
'''
nn2 = NeuralNet(layers=[13,10,1], learning_rate=0.01, iterations=500) # create the NN model
nn2.fit(x_train, y_train) # train the model

nn2.plot_loss()
print("Train accuracy of second network is {}".format(nn2.acc(y_train, train_pred)))
print("Test accuracy of second network is {}".format(nn2.acc(y_test, test_pred)))
