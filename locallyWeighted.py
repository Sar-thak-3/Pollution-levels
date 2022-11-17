# Locally weighted closed form regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = pd.read_csv("Train.csv")
x = x.values
y = x[:,-1]
x = x[:,:-1]
ones = np.ones((x.shape[0],1))
x_ = np.hstack((x,ones))
print(type(x))

def hypothesis(x,theta):
    return np.dot(x,theta)

def weight(xi,x,tau):
    w = np.mat(np.eye(xi.shape[0]))
    # print(w)
    m = xi.shape[0]
    denom = -2*tau*tau
    for i in range(m):
        num = np.dot((xi[i]-x),((xi[i]-x).T))
        w[i,i] = np.exp(num/denom)
        # print(w[i,i])
    return w

def locallyWeighted(x,y,w):
    firstPart = np.linalg.pinv(x.T*w*x)
    secondPart = (x.T)*w*y
    return firstPart*secondPart

def calculateAll(x,y,query,tau):
    ans = []
    for i in range(query.shape[0]):
        w = weight(x,query[i],tau)
        theta = locallyWeighted(np.mat(x),np.mat(y),np.mat(w))
        ans.append(hypothesis(query[i],np.array(theta)))
    return ans    

y = y.reshape((-1,1))
test = pd.read_csv("Test.csv")
test = test.values
ones = np.ones((test.shape[0],1))
test_ = np.hstack((test,ones))
print(test_)
ans = calculateAll(x_,y,test_,10)
answer = pd.DataFrame(np.array(ans) , columns=['target'])
answer.to_csv('answers.csv' ,index=False)