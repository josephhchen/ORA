import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from helper import y2indicator, getBinaryfer13Data, error_rate, \
                    init_weight_and_bias_NN, ReLU, softmax

class NNClass(object):
    def __init__(self):
        pass
    
    def train(self, X, Y, step_size=10e-7, epochs=10000, batch_size=100):
        print("Starting training...")
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        K = len(set(Y))
        
        Ytrain_ind = y2indicator(Y, K)
        Yvalid_ind = y2indicator(Yvalid, K)
        
        N, D = X.shape
        H1 = (D + K) // 2 
        H2 = H1 // 2  
        
        self.W1, self.b1 = init_weight_and_bias_NN(D, H1)
        self.W2, self.b2 = init_weight_and_bias_NN(H1, H2)
        self.W3, self.b3 = init_weight_and_bias_NN(H2, K)  
        
        train_costs = []
        valid_costs = []
        best_validation_error = 1
        
        n_batches = N // batch_size
        
        for i in range(epochs):
            X, Ytrain_ind = shuffle(X, Ytrain_ind) 
            for j in range(n_batches):
                Xbatch = X[j*batch_size:(j*batch_size + batch_size)]
                Ybatch = Ytrain_ind[j*batch_size:(j*batch_size + batch_size)]
                
                P_Y_given_X, Z1, A1, Z2, A2, Z3 = self.forward(Xbatch)
                
                dZ3 = P_Y_given_X - Ybatch
                self.W3 -= step_size * A2.T.dot(dZ3)
                self.b3 -= step_size * dZ3.sum(axis=0)
                
                dA2 = dZ3.dot(self.W3.T)
                dZ2 = dA2 * (Z2 > 0)
                self.W2 -= step_size * A1.T.dot(dZ2)
                self.b2 -= step_size * dZ2.sum(axis=0)

                dA1 = dZ2.dot(self.W2.T)
                dZ1 = dA1 * (Z1 > 0)
                self.W1 -= step_size * Xbatch.T.dot(dZ1)
                self.b1 -= step_size * dZ1.sum(axis=0)
            
            if i % 100 == 0 or i == epochs - 1:
                P_Y_given_X_valid, _, _, _, _, _ = self.forward(Xvalid)
                train_cost = self.cross_entropy(Ytrain_ind, self.forward(X)[0])
                valid_cost = self.cross_entropy(Yvalid_ind, P_Y_given_X_valid)
                train_costs.append(train_cost)
                valid_costs.append(valid_cost)
                validation_error = error_rate(Yvalid, self.predict(Xvalid))
                if validation_error < best_validation_error:
                    best_validation_error = validation_error
                print(f"Epoch: {i}/{epochs}, Training Cost: {train_cost:.4f}, Validation Cost: {valid_cost:.4f}, Validation Error: {validation_error:.4f}")
                
        print("Best validation error:", best_validation_error)
        
        plt.plot(train_costs, label='Training Costs')
        plt.plot(valid_costs, label='Validation Costs')
        plt.xlabel('Epochs (x100)')
        plt.ylabel('Cost')
        plt.title('Training and Validation Costs over Time')
        plt.legend()
        plt.show()
  
        
    def forward(self, X):
        Z1 = X.dot(self.W1) + self.b1
        A1 = ReLU(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        A2 = ReLU(Z2)
        Z3 = A2.dot(self.W3) + self.b3
        A3 = softmax(Z3)

        return A3, Z1, A1, Z2, A2, Z3

    def predict(self, X):
        P_Y_given_X, _, _, _ = self.forward(X) 
        return np.argmax(P_Y_given_X, axis=1)
        
    def classification_rate(self, Y, P):
        return np.mean(Y == P)

    def cross_entropy(self, T, pY):
        return -np.mean(T * np.log(pY))
    
X, Y = getBinaryfer13Data('voice_to_do_list_train.csv')
nnObj = NNClass()
nnObj.train(X, Y, epochs = 1000)

testX, testY = getBinaryfer13Data('voice_to_do_list_test.csv')
test_accuracy = nnObj.classification_rate(testX, testY)
print("Accuracy of test set is :", test_accuracy)
