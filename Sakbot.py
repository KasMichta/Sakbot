import numpy as np

test_set= np.array([[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,1,1]])
expected_out= np.array([[1,1,0,0,0]])
expected_out= expected_out.reshape(5,1)

np.random.seed(15)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
lr = 0.05

