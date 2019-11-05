# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:48:43 2019

@author: jairam
"""


import numpy as np


def train_validate_test_split(df, train_percent=.7, validate_percent=.1):
    
    split_1 = int(0.7 * len(df))
    split_2 = int(0.8 * len(df))
    dataset_train = df[:split_1]
    dataset_val = df[split_1:split_2]
    dataset_test = df[split_2:]
    return dataset_train,dataset_val,dataset_test


def sigmoid(s,beta):
    return 1/(1 + np.exp(-beta*s))
# 
def sigmoid_derv(s,beta):
    return beta*s * (1 - s)
# =============================================================================

#output layer actiavtion fns


def softmax(x, beta,derivative=False):
    if (derivative == True):
        return beta*x * (1 - x)
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

#hiiden layer activation functions

def tanh_fn(x):
    return ( np.exp(x) - np.exp(-x)) / ( np.exp(x) + np.exp(-x))

def tanh(x,beta, derivative=False):
    if (derivative == True):
        return beta*(1 - (x ** 2))
    return tanh_fn(beta*x)

def softplus(x, beta,derivative=False):
    if(derivative==True):
        return 1 / (1 + np.exp(-x))
    return np.log(1+np.exp(x))

def elu(x,delta, derivative=False):
    if(derivative==True):
        if(x>0):
            return 1
        else :
            return delta*np.exp(x)
    if(x>0):
        return x
    else :
        return delta*(np.exp(x)-1)
    

def relu(x, derivative=False):
    if(derivative==True):
        if(x>0):
            return 1
        else :
            return 0
    if(x>0):
        return x
    else :
        return 0

# =============================================================================
# def sigmoid(x, derivative=False,beta):
#     if (derivative == True):
#         return beta*x * (1 - x)
#     return 1 / (1 + np.exp(-beta*x))
# =============================================================================

    

def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
            
class_list = [0,2,3,6,7]
row_no = list()
fname = 'image_data_labels.txt'
f = open(fname,'r')
f2 = open('image_data_feat_dim60.txt')
X = list()
y1 = list()
y = list()
row = 1

for x in f.readlines():
    if int(x) in class_list:
        row_no.append(row)
        y1.append(int(x))
    row = row +1
f.close()

for val in y1:
    onehot = [0 for _ in range(len(class_list))]
    onehot[class_list.index(val)] = 1
    y.append(onehot)

row2 = 1
for line in f2.readlines():
    if row2 in row_no:
        X.append([float(x) for x in line.split()])
    row2 = row2 + 1
f2.close()

minmax = dataset_minmax(X)
normalize_dataset(X, minmax)
X=np.asarray(X)   
y=np.asarray(y)
#shuffling dataset
dataset = np.concatenate((X,y),axis = 1)
np.random.shuffle(dataset)
df=dataset.tolist()
dataset_train,dataset_val,dataset_test=train_validate_test_split(df,0.7,0.1)
#train
dataset_train=np.asarray(dataset_train)
X_train = dataset_train[:,0:60]
y_train = dataset_train[:,60:]
#val
dataset_val=np.asarray(dataset_val)
X_val = dataset_val[:,0:60]
y_val = dataset_val[:,60:]
#test
dataset_test=np.asarray(dataset_test)
X_test = dataset_test[:,0:60]
y_test = dataset_test[:,60:]


def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])  #logp=N*1
    loss = np.sum(logp)  #computed for all examples(N)
    return loss

class MyNN:
    def __init__(self, x, y,neurons):
        self.x = x
        self.lr = 0.1
        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        self.wIJ = np.random.randn(ip_dim, neurons)#between input,first hidden(d*neurons1)
        self.bh1 = np.zeros((1, neurons))#1*neurons1
# =============================================================================
#         self.wJM = np.random.randn(neurons, neurons)#between 2nd hidden,first hidden(neurons1*n2)
#         self.bh2 = np.zeros((1, neurons))#1*n2
# =============================================================================
        self.wJK= np.random.randn(neurons, op_dim)#between 2nd hidden,output(n2*K)
        self.bo = np.zeros((1, op_dim))#1*K
        self.y = y#N*K

    def feedforward(self,n,beta):
        #for first hidden layer
        x_n=self.x[n,:]
        x_n=x_n[np.newaxis,:]  #1*d
        ah1 = np.dot(x_n, self.wIJ) + self.bh1
        self.sh1 = sigmoid(ah1,beta)#for first hidden layer
        #2nd hidden layer
# =============================================================================
#         ah2 = np.dot(self.sh1, self.wJM) + self.bh2
#         self.sh2 = sigmoid(ah2)#1*nuerons2
# =============================================================================
        #for last layer
        ao = np.dot(self.sh1, self.wJK) + self.bo
        self.so = sigmoid(ao,beta)#for output layer  #1*K
# =============================================================================
#         ah1 = np.dot(self.x, self.wij) + self.bh1#N*nuerons1
#         self.sh1 = sigmoid(ah1)#for first hidden layer
#         ah2 = np.dot(self.sh1, self.wjm) + self.bh2
#         self.sh2 = sigmoid(ah2)#N*nuerons2
#         ao = np.dot(self.sh2, self.wmk) + self.bo
#         self.so = softmax(ao)#for output layer  #N*K
# =============================================================================
        
#    def local_grad(self,node,layer):
        
    def backprop_generalized_delta(self,n):
        #pattern mode
        #update of wmk
        x_n=self.x[n,:]
        x_n=x_n[np.newaxis,:]  #1*d
        y_n=self.y[n,:]
        y_n=y_n[np.newaxis,:]
        
        
        z3_delta = self.so - y_n # w3
        a3_delta = z3_delta * sigmoid_derv(self.so)
        z2_delta = np.dot(a3_delta, self.wMK.T)
        a2_delta = z2_delta * sigmoid_derv(self.sh2) # w2
        z1_delta = np.dot(a2_delta, self.wJM.T)
        a1_delta = z1_delta * sigmoid_derv(self.sh1) # w1
 
        self.wMK -= self.lr * np.dot(self.sh2.T, a3_delta)
        self.bo -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.wJM -= self.lr * np.dot(self.sh1.T, a2_delta)
        self.bh2 -= self.lr * np.sum(a2_delta, axis=0)
        self.wIJ -= self.lr * np.dot(x_n.T, a1_delta)
        self.bh1 -= self.lr * np.sum(a1_delta, axis=0)           
        
        
    def backprop_delta(self,n,beta):
        #pattern mode
        #update of wmk
        x_n=self.x[n,:]
        x_n=x_n[np.newaxis,:]  #1*d
        y_n=self.y[n,:]
        y_n=y_n[np.newaxis,:]
        
        
        z3_delta = self.so - y_n # w3
        a3_delta = z3_delta * sigmoid_derv(self.so,beta)
# =============================================================================
#         z2_delta = np.dot(a3_delta, self.wMK.T)
#         a2_delta = z2_delta * sigmoid_derv(self.sh2) # w2
# =============================================================================
        z1_delta = np.dot(a3_delta, self.wJK.T)
        a1_delta = z1_delta * sigmoid_derv(self.sh1,beta) # w1
 
        self.wJK -= self.lr * np.dot(self.sh1.T, a3_delta)
        self.bo -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
# =============================================================================
#         self.wJM -= self.lr * np.dot(self.sh1.T, a2_delta)
#         self.bh2 -= self.lr * np.sum(a2_delta, axis=0)
# =============================================================================
        self.wIJ -= self.lr * np.dot(x_n.T, a1_delta)
        self.bh1 -= self.lr * np.sum(a1_delta, axis=0)       
# =============================================================================
#         for k in range(self.so.shape[1]):
#             for m in range(self.sh2.shape[1]):
#                 wMK[m,k]+=self.lr*local_grad(n,k,3)*self.sh2[0,m]  #2=hidden layer no,3=output layer
#                 #n+1 is nth training ex(1 to N)
#         #update of wjm
#         for m in range(self.sh2.shape[1]):
#             for j in range(self.sh1.shape[1]):
#                 wJM[j,m]+=self.lr*local_grad(n,m,2)*self.sh1[0,j]
#         #update of wmk
#         for j in range(self.sh1.shape[1]):
#             for i in range(x_n.shape[1]):
#                 wIJ[i,j]+=self.lr*local_grad(n,j,1)*x_n[0,i]         
        
        
        
# =============================================================================
#         z3_delta = self.ao - self.y[n] # w3
#         a3_delta = z3_delta * sigmoid_derv(self.a3)
#         z2_delta = np.dot(a3_delta, self.w3.T)
#         a2_delta = z2_delta * sigmoid_derv(self.a2) # w2
#         z1_delta = np.dot(a2_delta, self.w2.T)
#         a1_delta = z1_delta * sigmoid_derv(self.a1) # w1
#  
#         self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
#         self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
#         self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
#         self.b2 -= self.lr * np.sum(a2_delta, axis=0)
#         self.w1 -= self.lr * np.dot(x_n.T, a1_delta)
#         self.b1 -= self.lr * np.sum(a1_delta, axis=0)
# =============================================================================
        

         
        
        
# =============================================================================
# =============================================================================
#         loss = error(self.a3, self.y)
#         print('Error :', loss)
#         a3_delta = cross_entropy(self.a3, self.y) # w3
#         z2_delta = np.dot(a3_delta, self.w3.T)
#         a2_delta = z2_delta * sigmoid_derv(self.a2) # w2
#         z1_delta = np.dot(a2_delta, self.w2.T)
#         a1_delta = z1_delta * sigmoid_derv(self.a1) # w1
# 
#         self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
#         self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
#         self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
#         self.b2 -= self.lr * np.sum(a2_delta, axis=0)
#         self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
#         self.b1 -= self.lr * np.sum(a1_delta, axis=0)
# =============================================================================

    def predict_feedforward(self,xx,yy,beta):
        x_n = xx[np.newaxis,:]
        ah1 = np.dot(x_n, self.wIJ) + self.bh1
        self.sh1 = sigmoid(ah1,beta)#for first hidden layer
        #2nd hidden layer
# =============================================================================
#         ah2 = np.dot(self.sh1, self.wJM) + self.bh2
#         self.sh2 = sigmoid(ah2)#1*nuerons2
# =============================================================================
        #for last layer
        ao = np.dot(self.sh1, self.wJK) + self.bo
        self.so = sigmoid(ao,beta)#for output layer  #1*K    
        rmse = (0.5*np.sum((yy-np.array(self.so))**2))
#        true_cls = yy.argmax()
#        pred_cls = self.so.argmax()
        return (rmse, self.so.argmax()) 
        
# =============================================================================
#     def predict(self, data):
#         self.x = data
#         self.predict_feedforward()
#         return self.so.argmax()
#     
#         
# =============================================================================
def get_acc(x, y,beta):
    acc = 0
    for xx,yy in zip(x, y):
        s = model.predict_feedforward(xx,yy,beta)[1]
        if s == np.argmax(yy):
            acc +=1
    return (acc/len(x)*100)	

best_beta=0
best_neurons=0	
max_val_acc=0
#check for convergence
for beta in [0.2,1,10]:
    for neurons in [20,70,120]:
        model = MyNN(X_train, np.array(y_train),neurons)
        sum_prev_error=0
        n_epochs=0        
        while(1):
            sum_error=0 
            threshold=0.01
            for n in range(X_train.shape[0]):  #pattern mode
                model.feedforward(n,beta)
                model.backprop_delta(n,beta)
            for xx,yy in zip(X_train, np.array(y_train)):
                sum_error+= model.predict_feedforward(xx,yy,beta)[0]     
            #convergence criterion
            if(abs(sum_error-sum_prev_error)<=threshold):
                print('parameters : beta={} and neurons={}'.format(beta,neurons))
                print('convergence has reached with difference of total error=',sum_error-sum_prev_error)
                print('no of epochs for convergence=',n_epochs)
                print("**********")
                break
            sum_prev_error=sum_error
            n_epochs+=1
#            print(sum_error)   
        print("Training accuracy for corresponding parameters: ",get_acc(X_train, y_train,beta))
        val_acc=get_acc(X_val, y_val,beta)
        print("validation accuracy for corresponding parameters: ", val_acc)  
        if(max_val_acc<val_acc):
            max_val_acc=val_acc
            best_beta=beta
            best_neurons=neurons
        print("\n\n")  
        print('-----------------------------------------------------------')
print('best parameters of base model with one hidden layer are:beta={} and neurons={}'.format(best_beta,best_neurons))        
# =============================================================================
#
# =============================================================================
		



#tupl = get_acc(x_train, np.array(y_train))

#print("Confusion matrix :")
#print(tupl[1])

