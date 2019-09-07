#!/usr/bin/env python
# coding: utf-8

# In[56]:

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# In[58]:

X = torch.randn(100, 1)*10
y = X + 2*torch.randn(100, 1)
plt.plot(X.numpy(), y.numpy(), 'o')
plt.show()


# In[59]:


# Inherit properties from the class nn.Module (parent class)
class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        pred = self.linear(x)
        return pred


# In[60]:


torch.manual_seed(3)


# In[61]:


model = LR(1,1)


# In[62]:


[w, b] = model.parameters()
w1 = w.item()
b1 = b.item()


# In[63]:


print(np.array([w.item(),b.item()]))


# In[64]:


print(list([w.item(),b.item()]))


# In[65]:


def get_params():
    return (w[0][0].item(), b[0].item())


# In[66]:


def plot_fit(title):
    plt.figure()
    plt.title = title
    w1, b1 = get_params()
    x1 = np.array([-30, 30])
    y1 = w1*x1 + b1
    plt.plot(x1, y1, 'r')
    plt.scatter(X, y)
    plt.show


# In[67]:


plot_fit('initial model')


# In[68]:


model


# In[69]:


print(list(model.parameters()))


# In[70]:


x = torch.tensor([[1.0], [7]])


# In[71]:


print(model.forward(x))


# In[72]:


#Manual computation of what is going on when putting forward the first part of the tensor
list(model.parameters())[0]*x[0,0] + list(model.parameters())[1] 


# In[73]:


#Manual computation of what is going on when putting forward the second part of the tensor
list(model.parameters())[0]*x[1,0] + list(model.parameters())[1] 


# In[74]:


# Define the loss function (minimum square error) and a stochastic gradient descent
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


# In[75]:


epoches = 100
losses = []
for i in range(epoches):
    y_pred = model.forward(X)
    loss = criterion(y_pred, y)
    print('epoch', i, 'loss: ', loss.item())
    
    losses.append(loss)
    # Reinitialize gradients because they accumulate when applying 
    optimizer.zero_grad()
    # Calculation of derivatives
    loss.backward() 
    # Update parameters
    optimizer.step()


# In[76]:

plt.figure()
plt.plot(range(epoches), losses)
plt.ylabel('Loss')
plt.xlabel('Epochs')


# In[77]:

plot_fit('Trained Model')


# In[79]:


# Linear Regresion finishes up here! following lines are dedicated to reshape and boolean/float comparisons in the numpy library

a = [0, 0, 1, 1, 2, 2, 3, 3]
type(a)


# In[90]:


a_reshape_np = np.reshape(a, (2,2,-1))
print(a_reshape_np)
print(type(a_reshape_np))    


# In[107]:


arr = np.array([1, 2, 3, 4, 5, 6, 7]) >5
print(arr)
print(type(arr)) 


# In[110]:


arr_Float = arr.astype(float)
print(arr_Float)
print(type(arr_Float)) 
print(arr)
print(type(arr)) 


# In[111]:


arr == arr_Float

