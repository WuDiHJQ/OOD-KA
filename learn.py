import torch
import torch.nn.functional as F


# a = torch.randn([100,3,64])

# b = a.permute(0,2,1)

# c = torch.matmul(a,b)

# print(c[50,0,1])
# print(c[50,1,0])
# v = (a[50,0,:] *a[50,1,:]).sum()
# print(v)

# a = torch.randn([100,3,64])
# v = a[0,0,:]
# s = torch.sqrt((v * v).sum())
# print(v/s)
# a = F.normalize(a, p=2, dim=2)
# print(a)

f0 = torch.randn([100,3,8,8])
f1 = torch.randn([100,3,8,8])
fs = torch.cat( [f0,f1], dim=1 )
N, C, H, W = fs.shape
fs = fs.view(N, C, -1)
fs = F.normalize(fs, p=2, dim=2)
sim_mat = torch.abs(torch.matmul(fs, fs.permute(0,2,1)))

sim_mat = torch.triu(sim_mat, 1)

loss_fea = sim_mat.sum() / (sim_mat != 0).sum()

loss = sim_mat.sum() / (N*C*(C-1) / 2)


print(sim_mat)
print(loss_fea)
print(loss)