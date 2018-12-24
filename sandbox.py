import numpy as np

x = np.random.randint(0,10,(3,10,15)) 
nx = x / np.linalg.norm(x,axis=0)
input_rays = np.stack((np.zeros(x.shape[1:]),np.zeros(x.shape[1:]),-1*np.ones(x.shape[1:])))


i = np.identity(3)[..., None, None]
r_matrices = i - 2 * np.einsum("ixy,jxy->ijxy", nx, nx)

#reflections = r_matrices @ input_rays

reflections = np.zeros(x.shape)
for ix0 in range(x.shape[1]):
    for iy0 in range(x.shape[2]):
        reflections[:,ix0,iy0] = r_matrices[:,:,ix0,iy0].T @ input_rays[:,ix0,iy0]

r2 = np.einsum("ijxy,ixy->jxy",r_matrices,input_rays)

#print(r2)
#print(reflections)
print(r2==reflections)
print(reflections.shape)