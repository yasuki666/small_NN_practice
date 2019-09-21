import torch
import time
if torch.cuda.is_available():
    a1 = time.time()
    x=torch.Tensor([[1,2],[2,3]])
    y=torch.Tensor([[1,2],[2,3]])
    x=x.cuda()
    y=y.cuda()
    Z=x+y
    b1 = time.time()
    print(Z)
    print(b1-a1)

a2 = time.time()
x = torch.Tensor([[1, 2], [2, 3]])
y = torch.Tensor([[1,2],[2,3]])
Z2=x+y
b2 = time.time()
print(Z2)
print(b2-a2)
print(time.time())