import torch

print(torch.version.cuda)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
my_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype = torch.float32, device = device,requires_grad= True )
print(my_tensor)


x = torch.empty(size= (3,3))
x= torch.zeros( size = (3,3))
x = torch.rand(size = (3,3))
x= torch.ones(size = (3,3))
x = torch.eye(3,3)
x = torch.arange(start=1,end = 10, step = 1)
x = torch.linspace(start=0.1 , end =1, steps=10)
x = torch.empty(size = (3,3)).normal_(mean = 0, std = 1)
x = torch.empty(size = (3,3)).uniform_(0,1)
x = torch.diag(torch.ones(3)) # == torch.eye(3,3)
print(x)


tensor = torch.arange(4)
print(tensor.bool())    # True/ Flase
print(tensor.short())   # int16
print(tensor.long())    #int64  (Important)
print(tensor.half())    #float16
print(tensor.float())   #float32 (Important)
print(tensor.double())  #float64

# Array to tensor
import numpy as np

np_array = np.zeros((3,3))
print(np_array)
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()
print(tensor)
print(np_array_back)


#Tensor Math

x = torch.tensor([1,2,3])
y = torch.tensor([2,3,4])

# add
z1 = torch.empty(3)
torch.add(x,y,out= z1 )
print(z1)
z2 = torch.add(x,y)
print(z2)
z3 = x +y
print(z3)
#SUb
z4 = x-y
print(z4)
#Division
z5 = torch.true_divide(x,y)
print(z5)
# inplace
t = torch.zeros(3)
t.add_(x)
print(t)
t = t+x
# Luy thua
z6 = x.pow(2)
z6 = x**2
print(z6)
#Nhan matran
x1 = torch.rand(3,5)
x2 = torch.rand(5,3)
x3 = torch.mm(x1,x2)
print(x3)
#exp
matrix_exp = torch.rand(3,3)
print(matrix_exp.matrix_power(3))

#element wise mult
z = x*y
print(z)

# dot product

z7 = torch.dot(x,y)
print(z7)
# Batch maxtrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch,n,m))
print(tensor1.shape)
tensor2 = torch.rand((batch,m,p))
print(tensor2.shape)
out_bmm = torch.bmm(tensor1,tensor2)
print(out_bmm.shape)

# Broadcasting
x1 = torch.rand((5,5))
print(x1)
print('--------------------')
x2 = torch.rand(1,5)

print(x2)
print('--------------')
z = x1 - x2
z = x1 ** x2
print(z)
# Other useful
print("-----------------------------------")
sum_x = torch.sum(x,dim = 0)
print(sum_x)
values,idx = torch.max(x,dim = 0)
print(values,idx)
values,idx = torch.min(x,dim = 0)
print(values,idx)
abs_x = torch.abs(x)
print(abs_x)
z = torch.argmax(x,dim = 0)
print(z)
z = torch.argmin(x,dim = 0)
print(z)
mean_x = torch.mean(x.float(),dim = 0)
print(mean_x)
z = torch.eq(x,y)
print(z)
sorted,idx = torch.sort(x,dim = 0,descending=False)
print(sorted,idx)
z =torch.clamp(x,min = 1)
print(z)

# Indexing

batch_size = 10
features = 25

x = torch.rand((batch_size,features))

print(x[0].shape)
print(x[:,0].shape)
print(x[2,1:10])

x = torch.arange(10)
indices = [2,5,8]
print(x[indices])

x = torch.rand([3,5])
print(x)
rows = torch.tensor([1,0])
print(rows)
cols = torch.tensor([4,0])
print(cols)
print(x[rows,cols])
x = torch.arange(10)
print(x[(x<2)|(x>8)])
print(x[x.remainder(2) == 0])
print(torch.where(x>5,x,x*2))
print(torch.tensor([1,2,3,4,1,2,3]).unique())
print(x.ndimension())
print(x.numel())

#
x = torch.arange(9)
print(x)
x_3x3 = x.view(3,3)
print(x_3x3)
x_3x3 = x.reshape(3,3)
print(x_3x3)
y = x_3x3.t()
print(y.contiguous().view(9)) # tensor([0, 3, 6, 1, 4, 7, 2, 5, 8])

x1 = torch.rand((3,5))
x2 = torch.rand((3,5))
print(torch.cat((x1,x2),dim = 0).shape)
print(torch.cat((x1,x2),dim = 1).shape)

z = x1.view(-1)
print(z.shape)

batch =64
x = torch.rand((64,3,5))
z = x.view(batch,-1)
print(z.shape)
z = x.permute(0,2,1)
print(z.shape)
x = torch.arange(10).unsqueeze(0).unsqueeze(1)
z = x.unsqueeze(1)
print(z.shape)

