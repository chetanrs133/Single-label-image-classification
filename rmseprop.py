
import numpy as np
from sklearn.model_selection import train_test_split


#actiavtion functions
# =============================================================================
# def sigmoid(s):
#     return 1/(1 + np.exp(-s))
# 
# def sigmoid_derv(s):
#     return s * (1 - s)
# =============================================================================

#output layer actiavtion fns


# =============================================================================
# def softmax(x, derivative=False,beta):
#     if (derivative == True):
#         return beta*x * (1 - x)
#     exps = np.exp(s - np.max(s, axis=1, keepdims=True))
#     return exps/np.sum(exps, axis=1, keepdims=True)
# 
# #hiiden layer activation functions
# 
# def tanh_fn(x):
#     return ( np.exp(x) - np.exp(-x)) / ( np.exp(x) + np.exp(-x))
# 
# def tanh(x, derivative=False,beta):
#     if (derivative == True):
#         return beta*(1 - (x ** 2))
#     return tanh_fn(beta*x)
# 
# def softplus(x, derivative=False,beta):
#     if(derivative==True):
#         return 1 / (1 + np.exp(-x))
#     return np.log(1+np.exp(x))
# 
# def elu(x, derivative=False,delta):
#     if(derivative==True):
#         if(x>0):
#             return 1
#         else :
#             return delta*np.exp(x)
#     if(x>0):
#         return x
#     else :
#         return delta*(np.exp(x)-1)
#     
# 
# def relu(x, derivative=False):
#     if(derivative==True):
#         if(x>0):
#             return 1
#         else :
#             return 0
#     if(x>0):
#         return x
#     else :
#         return 0
# 
# =============================================================================
def sigmoid(x, derivative=False,beta = 1):
    if (derivative == True):
        return beta*x * (1 - x)
    return 1 / (1 + np.exp(-beta*x))

    

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

dataset = np.concatenate((X,y),axis = 1)
np.random.shuffle(dataset)
X = dataset[:,0:60]
y = dataset[:,60:]

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=20)





def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])  #logp=N*1
    loss = np.sum(logp)  #computed for all examples(N)
    return loss

class MyNN:
    def __init__(self, x, y):
        self.x = x
        neurons = 80
        self.lr = 0.08
        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        self.wIJ = np.random.randn(ip_dim, neurons)#between input,first hidden(d*neurons1)
        self.bh1 = np.zeros((1, neurons))#1*neurons1
        self.wJM = np.random.randn(neurons, neurons)#between 2nd hidden,first hidden(neurons1*n2)
        self.bh2 = np.zeros((1, neurons))#1*n2
        self.wMK= np.random.randn(neurons, op_dim)#between 2nd hidden,output(n2*K)
        self.bo = np.zeros((1, op_dim))#1*K
        self.y = y#N*K

    def feedforward(self,n):
        #for first hidden layer
        x_n=self.x[n,:]
        x_n=x_n[np.newaxis,:]  #1*d
        ah1 = np.dot(x_n, self.wIJ) + self.bh1
        self.sh1 = sigmoid(ah1)#for first hidden layer
        #2nd hidden layer
        ah2 = np.dot(self.sh1, self.wJM) + self.bh2
        self.sh2 = sigmoid(ah2)#1*nuerons2
        #for last layer
        ao = np.dot(self.sh2, self.wMK) + self.bo
        self.so = sigmoid(ao)#for output layer  #1*K
# =============================================================================
#         ah1 = np.dot(self.x, self.wij) + self.bh1#N*nuerons1
#         self.sh1 = sigmoid(ah1)#for first hidden layer
#         ah2 = np.dot(self.sh1, self.wjm) + self.bh2
#         self.sh2 = sigmoid(ah2)#N*nuerons2
#         ao = np.dot(self.sh2, self.wmk) + self.bo
#         self.so = softmax(ao)#for output layer  #N*K
# =============================================================================
        
#    def local_grad(self,node,layer):
        
    def update_weight(self,s,str2,n,m,g_m,epsilon = 0.0000000000000000000000000000000000000001, L = 1, rho = 0.1, rho1 = 0.9, rho2 = 0.999):
        
        if s == "AdaGrad":
            #print(self.g[n][s]['sum'][str2])
            r_w = np.zeros((self.g)[n][s]['sum'][str2].shape)
            r_w = self.g[n][s]['sum_array'][str2][m]
# =============================================================================
#             print(r_w)
#             print(g_m**2)
# =============================================================================
# =============================================================================
#             for k in range(m):
#                 r_w+=(self.g)[n][k]**2
# =============================================================================
# =============================================================================
#                 for i in range(self.g[n][0].shape[0]):
#                     for j in range(self.g[n][0].shape[1]):
#                         r_w[i][j] +=(self.g[n][k][i][j])**2
# =============================================================================
            r_w = np.sqrt(r_w)
            r_w = r_w + epsilon
            return np.divide(g_m,r_w)
        
        if s == "Delta":
            return g_m
        
        if s == "rmsprop":
            r_w = np.zeros((self.g)[n][s]['sum'].shape)
            if m>=L:
                r_w = 0.5*(self.g[n][s]['sum_array'][m-1] - self.g[n][s]['sum_array'][m-L])*rho
            r_w += (1-rho)*(g_m**2)           
            r_w = np.sqrt(r_w)
            r_w = r_w + epsilon
            return np.divide(g_m,r_w)
        
        if s == 'AdaDelta':
            firstorder = (self.g[n][s]['firstorder'][m])/(1 - (rho1**m))
            secondorder = (self.g[n][s]['secondorder'][m])/(1 - (rho2**m))
            secondorder = np.sqrt(secondorder)
            secondorder+=epsilon
            return np.divide(firstorder,secondorder)
            
        
    def backprop(self,n,s,epsilon = 0.01, rho = 0.1,rho1= 0.9, rho2 = 0.999):
        #pattern mode
        #update of wmk
        
        if n == 0:
            self.g =dict()
            self.g[3] = dict()
            self.g[2] = dict()
            self.g[1] = dict()

            #data structures for storing sum and moments for delta update rule
            #not required
            (self.g)[3]['Delta'] = dict()
            (self.g)[2]['Delta'] = dict()
            (self.g)[1]['Delta'] = dict()


            (self.g)[3]['Delta']['sum'] = np.zeros(self.wMK.shape)
            (self.g)[2]['Delta']['sum'] = np.zeros(self.wJM.shape)
            (self.g)[1]['Delta']['sum'] = np.zeros(self.wIJ.shape)
            self.g[3]['Delta']['sum_array'] = list()
            self.g[2]['Delta']['sum_array'] = list()            
            self.g[1]['Delta']['sum_array'] = list()
            
            #data structures for storing sum and moments for AdaGrad update rule
            (self.g)[3]['AdaGrad'] = dict()
            (self.g)[2]['AdaGrad'] = dict()
            (self.g)[1]['AdaGrad'] = dict()

            (self.g)[3]['AdaGrad']['sum'] = dict()
            (self.g)[2]['AdaGrad']['sum'] = dict()
            (self.g)[1]['AdaGrad']['sum'] = dict()

            (self.g)[3]['AdaGrad']['sum_array'] = dict()
            (self.g)[2]['AdaGrad']['sum_array'] = dict()
            (self.g)[1]['AdaGrad']['sum_array'] = dict()

            (self.g)[3]['AdaGrad']['sum']['weights'] = np.zeros(self.wMK.shape)
            (self.g)[2]['AdaGrad']['sum']['weights'] = np.zeros(self.wJM.shape)
            (self.g)[1]['AdaGrad']['sum']['weights'] = np.zeros(self.wIJ.shape)
            self.g[3]['AdaGrad']['sum_array']['weights'] = list()
            self.g[2]['AdaGrad']['sum_array']['weights'] = list()            
            self.g[1]['AdaGrad']['sum_array']['weights'] = list()


            (self.g)[3]['AdaGrad']['sum']['bias'] = np.zeros(self.bo.shape)
            (self.g)[2]['AdaGrad']['sum']['bias'] = np.zeros(self.bh2.shape)
            (self.g)[1]['AdaGrad']['sum']['bias'] = np.zeros(self.bh1.shape)
            self.g[3]['AdaGrad']['sum_array']['bias'] = list()
            self.g[2]['AdaGrad']['sum_array']['bias'] = list()            
            self.g[1]['AdaGrad']['sum_array']['bias'] = list()
                        
            #data structures for storing sum and moments for rmsprop update rule
            (self.g)[3]['rmsprop'] = dict()
            (self.g)[2]['rmsprop'] = dict()
            (self.g)[1]['rmsprop'] = dict()


            (self.g)[3]['rmsprop']['sum'] = np.zeros(self.wMK.shape)
            (self.g)[2]['rmsprop']['sum'] = np.zeros(self.wJM.shape)
            (self.g)[1]['rmsprop']['sum'] = np.zeros(self.wIJ.shape)
            self.g[3]['rmsprop']['sum_array'] = list()
            self.g[2]['rmsprop']['sum_array'] = list()            
            self.g[1]['rmsprop']['sum_array'] = list()
            

            #data structures for storing sum and moments for adam update rule
            (self.g)[3]['adam'] = dict()
            (self.g)[2]['adam'] = dict()
            (self.g)[1]['adam'] = dict()


            self.g[3]['adam']['firstorder'] = list()
            self.g[2]['adam']['firstorder'] = list()
            self.g[1]['adam']['firstorder'] = list()
            self.g[3]['adam']['secondorder'] = list()
            self.g[2]['adam']['secondorder'] = list()            
            self.g[1]['adam']['secondorder'] = list()


        x_n=self.x[n,:]
        x_n=x_n[np.newaxis,:]  #1*d
        y_n=self.y[n,:]
        y_n=y_n[np.newaxis,:]


        
        z3_delta = self.so - y_n # w3
        a3_delta = z3_delta * sigmoid(self.so, derivative = True)
        z2_delta = np.dot(a3_delta, self.wMK.T)
        a2_delta = z2_delta * sigmoid(self.sh2, derivative = True) # w2
        z1_delta = np.dot(a2_delta, self.wJM.T)
        a1_delta = z1_delta * sigmoid(self.sh1, derivative = True) # w1
 
        g3_m = np.dot(self.sh2.T, a3_delta)
        g2_m = np.dot(self.sh1.T, a2_delta)
        g1_m = np.dot(x_n.T, a1_delta)
        
        g3b_m = np.sum(a3_delta, axis=0, keepdims=True)
        g2b_m = np.sum(a2_delta, axis=0)
        g1b_m = np.sum(a1_delta, axis=0)        
        
        if s == 'AdaGrad':
            self.g[3][s]['sum']['weights']+=g3_m**2
            self.g[2][s]['sum']['weights']+=g2_m**2
            self.g[1][s]['sum']['weights']+=g1_m**2
            self.g[3][s]['sum_array']['weights'].append(self.g[3][s]['sum']['weights'])
            self.g[2][s]['sum_array']['weights'].append(self.g[2][s]['sum']['weights'])        
            self.g[1][s]['sum_array']['weights'].append(self.g[1][s]['sum']['weights']) 

            self.g[3][s]['sum']['bias']+=g3b_m**2
            self.g[2][s]['sum']['bias']+=g2b_m**2
            self.g[1][s]['sum']['bias']+=g1b_m**2
            self.g[3][s]['sum_array']['bias'].append(self.g[3][s]['sum']['bias'])
            self.g[2][s]['sum_array']['bias'].append(self.g[2][s]['sum']['bias'])        
            self.g[1][s]['sum_array']['bias'].append(self.g[1][s]['sum']['bias']) 
            
        if s == 'rmsprop':
            self.g[3][s]['sum']+=g3_m**2
            self.g[2][s]['sum']+=g2_m**2
            self.g[1][s]['sum']+=g1_m**2
            self.g[3][s]['sum_array'].append(self.g[3]['sum'])
            self.g[2][s]['sum_array'].append(self.g[2]['sum'])        
            self.g[1][s]['sum_array'].append(self.g[1]['sum']) 
            
        if s == 'adam':
            if n == 0:
                firstorder3 = np.zeros(self.wMK.shape)
                firstorder2 = np.zeros(self.wJM.shape)
                firstorder1 = np.zeros(self.wIJ.shape)
            else:
                firstorder3  = rho1*self.g[3][s]['firstorder'][n-1] + (1-rho1)*g3_m
                firstorder2 = rho1*self.g[2][s]['firstorder'][n-1] + (1-rho1)*g2_m
                firstorder1 = rho1*self.g[1][s]['firstorder'][n-1] + (1-rho1)*g1_m
            self.g[3][s]['firstorder'].append(firstorder3)
            self.g[2][s]['firstorder'].append(firstorder2)        
            self.g[1][s]['firstorder'].append(firstorder1) 
            if n == 0:
                secondorder3 = np.zeros(self.wMK.shape)
                secondorder2 = np.zeros(self.wJM.shape)
                secondorder1 = np.zeros(self.wIJ.shape)
            else:
                secondorder3  = rho2*self.g[3][s]['secondorder'][n-1] + (1-rho2)*(g3_m**2)
                secondorder2 = rho2*self.g[2][s]['secondorder'][n-1] + (1-rho2)*(g2_m**2)
                secondorder1 = rho2*self.g[1][s]['secondorder'][n-1] + (1-rho2)*(g1_m**2)
            self.g[3][s]['secondorder'].append(secondorder3)
            self.g[2][s]['secondorder'].append(secondorder2)        
            self.g[1][s]['secondorder'].append(secondorder1) 
        
        
        
        
        self.wMK -= self.lr * self.update_weight(s,'weights',3,n,g3_m)
        self.bo -= self.lr * self.update_weight(s,'bias',3,n,g3b_m)
        self.wJM -= self.lr * self.update_weight(s,'weights',2,n,g2_m)
        self.bh2 -= self.lr * self.update_weight(s,'bias',2,n,g2b_m)
        self.wIJ -= self.lr * self.update_weight(s,'weights',1,n,g1_m)
        self.bh1 -= self.lr * self.update_weight(s,'bias',1,n,g1b_m)
         
        
        


    def predict_feedforward(self,xx,yy):
        x_n = xx[np.newaxis,:]
        ah1 = np.dot(x_n, self.wIJ) + self.bh1
        self.sh1 = sigmoid(ah1)#for first hidden layer
        #2nd hidden layer
        ah2 = np.dot(self.sh1, self.wJM) + self.bh2
        self.sh2 = sigmoid(ah2)#1*nuerons2
        #for last layer
        ao = np.dot(self.sh2, self.wMK) + self.bo
        self.so = sigmoid(ao)#for output layer  #1*K    
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
			
model = MyNN(x_train, np.array(y_train))
sum_prev_error=0
n_epochs=0
while(1):
    sum_error=0 
    threshold=0.0005
    for n in range(x_train.shape[0]):  #pattern mode
        model.feedforward(n)
        model.backprop(n,"AdaGrad")
        
    for xx,yy in zip(x_train, np.array(y_train)):
        sum_error+= model.predict_feedforward(xx,yy)[0]     
    #convergence criterion
    if(abs(sum_error-sum_prev_error)<=threshold):
        print('convergence has reached with difference of total error=',sum_error-sum_prev_error)
        print('no of epochs for convergence=',n_epochs)
        break
    sum_prev_error=sum_error
    n_epochs+=1
    print(sum_error)   
# =============================================================================
#
# =============================================================================
		
def get_acc(x, y):
    acc = 0
    for xx,yy in zip(x, y):
        s = model.predict_feedforward(xx,yy)[1]
        if s == np.argmax(yy):
            acc +=1
    return (acc/len(x)*100)


#tupl = get_acc(x_train, np.array(y_train))
print("Training accuracy : ",get_acc(x_train, np.array(y_train) ))
print("Test accuracy : ", get_acc(x_val, np.array(y_val)))
#print("Confusion matrix :")
#print(tupl[1])

