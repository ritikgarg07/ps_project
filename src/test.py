import h5py

file = '/workspaces/ps_project/data/test.h5'
f = h5py.File(file, 'r')
ip_group = f['/ip']
print(ip_group.keys())