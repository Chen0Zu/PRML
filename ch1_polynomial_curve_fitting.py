import numpy as np
import matplotlib.pyplot as plt

def polynomial(X, M):
    n = X.shape[0]
    new_X = np.ones([n, M+1])
    for i in range(1,M+1):
        new_X[:,[i]] = new_X[:,[i-1]] * X
    return new_X

def model(X,y):
    n,d = X.shape
    w = np.zeros([d,1])
    w = np.linalg.inv(X.T@X)@X.T@y
    return w

def model_reg(X,y,alpha):
    n,d = X.shape
    w = np.zeros([d,1])
    w = np.linalg.inv(X.T@X+alpha*np.eye(d))@X.T@y
    return w

def predict(X,y,w):
    n,d = X.shape
    predict_y = X@w
    loss = 0.5*sum((predict_y-y)**2)
    rms_err = np.sqrt(2*loss/n)
    return predict_y, rms_err

#x = np.array([[1],[2],[3]])
#y= polynomial(x, 2)

x = np.linspace(0,1,100).reshape(-1,1)
y = np.sin(2*np.pi*x)

noise = np.random.normal(loc = 0, scale = 0.2, size = 10).reshape(-1,1)
z = np.random.uniform(size = 10).reshape(-1,1)
zt = np.sin(2*np.pi*z) + noise
plt.figure()
plt.plot(x,y)
plt.scatter(z, zt)

plt.figure(figsize=(13,10))
plt.subplot(221)
M = 0
w = model(polynomial(z,M),zt)
predict_y,rms_err = predict(polynomial(x,M), y, w)
plt.plot(x,y)
plt.scatter(z, zt)
plt.plot(x,predict_y)
plt.ylim(-1.5,1.5)
plt.subplot(222)
M = 1
w = model(polynomial(z,M),zt)
predict_y = polynomial(x,M)@w
plt.plot(x,y)
plt.scatter(z, zt)
plt.plot(x,predict_y)
plt.ylim(-1.5,1.5)
plt.subplot(223)
M = 3
w = model(polynomial(z,M),zt)
predict_y = polynomial(x,M)@w
plt.plot(x,y)
plt.scatter(z, zt)
plt.plot(x,predict_y)
plt.ylim(-1.5,1.5)
plt.subplot(224)
M = 9
w = model(polynomial(z,M),zt)
predict_y,rms_err = predict(polynomial(x,M), y, w)
plt.plot(x,y)
plt.scatter(z, zt)
plt.plot(x,predict_y)
plt.ylim(-1.5,1.5)

# =============================================================================
M = 10
train_error = np.zeros([M,1])
test_error = np.zeros([M,1])
for m in range(M):
    w = model(polynomial(z,m), zt)
    predict_y, train_error[m] = predict(polynomial(z,m), zt, w)
    predict_y, test_error[m] = predict(polynomial(x,m), y, w)
    
plt.figure()
l1, = plt.plot(range(M), train_error, 'o-')
l2, = plt.plot(range(M), test_error, 'o-')
plt.ylim(0,2)
plt.legend(handles = [l1, l2,], labels = ['Training', 'Test'], loc = 'best')

# =============================================================================
noise = np.random.normal(loc = 0, scale = 0.2, size = 15).reshape(-1,1)
z1 = np.random.uniform(size = 15).reshape(-1,1)
zt1 = np.sin(2*np.pi*z1) + noise
plt.figure(figsize=(14,8))
plt.subplot(121)
plt.plot(x,y)
plt.scatter(z1, zt1)
plt.plot(x, polynomial(x, 9)@model(polynomial(z1,9), zt1))
plt.ylim(-1.5, 1.5)

noise = np.random.normal(loc = 0, scale = 0.2, size = 100).reshape(-1,1)
z2 = np.random.uniform(size = 100).reshape(-1,1)
zt2 = np.sin(2*np.pi*z2) + noise
plt.subplot(122)
plt.plot(x,y)
plt.scatter(z2, zt2)
plt.plot(x, polynomial(x, 9)@model(polynomial(z2,9), zt2))
plt.ylim(-1.5, 1.5)

# =============================================================================
plt.figure(figsize=(14,7))
plt.subplot(121)
plt.plot(x,y)
plt.scatter(z, zt)
w1 = model_reg(polynomial(z,9), zt, 0.001)
plt.plot(x, polynomial(x, 9)@w1)
plt.ylim(-1.5, 1.5)
plt.subplot(122)
plt.plot(x,y)
plt.scatter(z, zt)
w2 = model_reg(polynomial(z,9), zt, 1)
plt.plot(x, polynomial(x, 9)@w2)
plt.ylim(-1.5, 1.5)