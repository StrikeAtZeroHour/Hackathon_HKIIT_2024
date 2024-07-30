import torch
from matplotlib import pyplot
#x是時間，y是房價，利用虛擬的數據、產生最準確的房價推測路徑。
x=torch.linspace(0,100,101)#製造101個數值，裏面的數值為0到100順序排列
y=x+torch.randn(101)*10#torch.randn(101)製造101個-1至+1的數值

train_dataX=x[:-20]
train_dataY=y[:-20]
test_dataX=x[-20:]
test_dataY=y[-20:]


#y=ax+c, find  the best a and b value
a= torch.rand(1,requires_grad=True)#打開梯度跟蹤功能
b= torch.rand(1,requires_grad=True)
learning_rate=0.001#學習率太高會適得其反，反而增加誤差
for iter in range(100):
    

    predictY=a*train_dataX+b
    error=abs(predictY-train_dataY).mean()
    error.backward()
    a.data.add_(-learning_rate*a.grad)#update the variable by the gradient value, you can only use data.add_() to edit value as PyTorch不允許對需要梯度跟蹤的張量進行原地修改操作,因為這樣會破壞之前計算的梯度信息,導致反向傳播過程出錯。
    b.data.add_(-learning_rate*b.grad)
    a.grad.zero_()#歸零 gradient，否則會積纍到之後每一次計算，導致計算錯誤
    b.grad.zero_()


pyplot.scatter(x.numpy(),y.numpy(),s=10)

pyplot.plot(x.numpy(),(a*x+b).detach().numpy())#.detach() to 取消梯度跟蹤 as PyTorch不允許對需要梯度跟蹤的張量進行原地修改操作,因為這樣會破壞之前計算的梯度信息,導致反向傳播過程出錯。
pyplot.show()