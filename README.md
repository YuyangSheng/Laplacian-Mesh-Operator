# Laplacian-Mesh-Operator
This repository  contains operation for discrete curvature, spectral meshes and laplacian mesh smoothing.
* Implementation of discrete curvature, including mean curvature, gaussian curvature and Laplace-Beltrami mean curvature.
* Mesh reconstruction with different size of eigenvectors.
* Laplacian Mesh Smoothing using Explicit and Implicit methods.
* Test the smoothing method on denoising tasks with synthetic gaussian noise.

## Requirements
The following is the version of several libraries to be used for this task. The python version is 3.9.12.
* trimesh == 3.18.0
* matplotlib == 3.5.1
* open3d == 0.16.0
* scipy == 1.10.0
* numpy == 1.23.5

## To start with
You can try `demo.ipynb` to easily get start with the code.

## Visualization
**Bumpy-cube with Laplace-Beltrami Mean Curvatures**
![Laplace-Beltrami mean curvature](https://github.com/Ilvecoding0912/Laplacian-Mesh-Operator/blob/main/results_img/bumpy-cube%20with%20cotangent%20mean%20curvatures.png)

**Mesh Reconstruction with Armadillo Object**
![Mesh reconstruction with armadillo](https://github.com/Ilvecoding0912/Laplacian-Mesh-Operator/blob/main/results_img/reconstruction%20of%20armadillo%20object.png)

**Mesh Smoothing using Plane Object**
![Mesh smoothing](https://github.com/Ilvecoding0912/Laplacian-Mesh-Operator/blob/main/results_img/mesh%20smoothing%20with%20plane%20object.png)
