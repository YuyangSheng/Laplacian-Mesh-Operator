import numpy as np
import trimesh
import matplotlib
import matplotlib.cm as cm
import open3d as o3d
import scipy
import scipy.sparse.linalg as la
from scipy.sparse.linalg import spsolve, eigsh
import math


class DiscreteCurvature:
    def __init__(self, tm):
        self.tm = tm
        self.N = len(tm.vertices) # length of set of vertices 

    
    def get_boundary_vertices(self):
        '''
        Return the indices of boundary vertices of the trimesh object
        '''

        unique_edges = self.tm.edges[trimesh.grouping.group_rows(self.tm.edges_sorted, require_count=1)]
        boundary_vertices = np.unique(unique_edges.flatten())

        return boundary_vertices

    def exclude_boundary_indices(self):
        '''
        Return all the indices of vertices excluding boundary vertices of the trimesh object
        '''
        # create indices of all vertices of the trimesh object
        vertex_indices = np.linspace(0, self.N-1, self.N)

        # get the indices of boundary vertices
        idx_boundary = self.get_boundary_vertices()

        # mark out boundary vertices as 0, others as 1
        bool_boundary = np.ones_like(vertex_indices)
        bool_boundary[idx_boundary] = 0

        # convert 1, 0 to 'True' or 'False'
        bool_boundary = np.where(bool_boundary == 1, True, False)

        # exclude boundary indices of vertices
        indices_exl_bound = vertex_indices[bool_boundary]

        return indices_exl_bound.astype('int'), vertex_indices.astype('int')


    def get_angles_and_areas(self, idx_current_vertex):
        '''
        Calculate the sum of all angles of A(vi) for current vertex
        '''

        # all the indices of edges which contain current vertex
        idx_edges = np.where(self.tm.edges == idx_current_vertex)[0]
        edges = self.tm.edges[idx_edges]


        # find out where current vertex is at the last column of the edges array
        current_vertex_end = (np.where(edges == idx_current_vertex)[1] == 1)

        # change the order -> current vertex at the first column of the edges array
        edges[current_vertex_end] = edges[current_vertex_end][:, [1, 0]]

        # restrict the order of vector == (current vertex - its neighbor)
        edge_vectors = self.tm.vertices[edges[:, 0]] - self.tm.vertices[edges[:, 1]] 


        # get odd & even row index respectively (cos two adjacent edges are on a face)
        even_vec, odd_vec = edge_vectors[0::2], edge_vectors[1::2]
        angles = np.diag(even_vec @ odd_vec.T) / (np.linalg.norm(even_vec, axis=1) * np.linalg.norm(odd_vec, axis=1))
        
        # compute the sum of all the angles
        sum_angles = np.sum(np.arccos(angles))
        
        # calculate the areas of the triangles -> (len(neighbors), 1)
        areas = np.linalg.norm(np.cross(even_vec, odd_vec), axis=1) / 2.

        # 1/3 sum of the areas
        sum_areas = areas.sum() / 3
        return sum_angles, sum_areas

    
    def compute_areas(self, indices):
        '''
        Return the area of neighbor faces of indices of vertices
        '''
        # get the indices of neighbor faces of vertices (non-neighbor -> -1)
        face_indices = self.tm.vertex_faces[indices]

        # add 0 to the end of the array (non-neighbor[indices] -> 0)
        faces_area = np.hstack([self.tm.area_faces, 0])
        
        area = faces_area[face_indices]

        return area
    
    def compute_cot_angle(self, vec1, vec2):
        '''
        Compute cotangent value of angles between two vectors
        '''
        cos_angle = vec1 @ vec2 / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cot_angle = 1 / math.tan(np.arccos(cos_angle))

        return cot_angle

    def compute_mean_curvature(self):
        '''
        Return the mean curvature of the trimesh object
        '''
        # construct m matrix and c matrix
        
        m_mat, c_mat = np.zeros((self.N, self.N)), np.zeros((self.N, self.N))
        
        for i in range(self.N):
            num_neighbors = len(self.tm.vertex_neighbors[i])
            m_mat[i, i] = num_neighbors

            c_mat[i, self.tm.vertex_neighbors[i]] = 1
            c_mat[i, i] = -num_neighbors
        
        m_mat, c_mat = scipy.sparse.csc_matrix(m_mat), scipy.sparse.csc_matrix(c_mat)
        laplacian_mat = spsolve(m_mat, c_mat)
        mean_curvature = 0.5 * np.linalg.norm(laplacian_mat @ self.tm.vertices, axis=1)

        return mean_curvature
    

    def compute_gaussian_curvature(self):
        '''
        Return the gaussian curvature of a trimesh object
        '''
    
        # create gaussian curvature array
        gaussian_curvature = np.zeros(self.N)
        
        for i in range(self.N):
            idx = int(i)
            # calculate the sum of angles and areas
            sum_angles, sum_areas = self.get_angles_and_areas(idx)
            gaussian_curvature[idx] = (2 * np.pi - sum_angles) / sum_areas
        
        return gaussian_curvature

    
    def compute_non_uniform_laplace(self):
        '''
        Calculate non-uniform Laplace (Laplace-Beltrami) for mean curvature
        '''

        # construct m matrix and c matrix
        m_mat, c_mat = np.zeros((self.N, self.N)), np.zeros((self.N, self.N))

        for idx in range(self.N):
            '''
                            angle_vertex1
                                 / \ 
                                /   \ 
                               /     \      
                           idx ——————— neighbor
                               \     /
                                \   /
                                 \ /
                              angle_vertex2
            '''

            # indices of current vertex's neighbors
            vertex_neighbors = self.tm.vertex_neighbors[idx]

            # sum of calculated cot angles
            sum_angles = 0

            for neighbor in vertex_neighbors:
                # sum of calculated cot angles for each neighbor of current vertex
                sum_cot_angles = 0

                # find the common vertices of current vertex's neighbor and its neighbor's neighbors
                v_nn = self.tm.vertex_neighbors[neighbor]
                angle_vertices = set(v_nn) & set(vertex_neighbors)
                
                # find the correct neighbor who can consist a A(vi) with current vertex vi
             
                # calculate the sum of cotangents of two angles for neighbor vertex
                for i in range(len(angle_vertices)):
                    angle_vertex = angle_vertices.pop()
                        
                    cot_angle = self.compute_cot_angle(self.tm.vertices[angle_vertex]-self.tm.vertices[idx], self.tm.vertices[angle_vertex]-self.tm.vertices[neighbor])
                    sum_cot_angles += cot_angle
                
                c_mat[idx, neighbor] = sum_cot_angles
                sum_angles += sum_cot_angles
            
            c_mat[idx, idx] = -sum_angles

            # get 1/3 sum of areas of triangles
            _, sum_areas = self.get_angles_and_areas(idx)
            m_mat[idx, idx] = sum_areas * 2

        m_mat, c_mat = scipy.sparse.csc_matrix(m_mat), scipy.sparse.csc_matrix(c_mat)
        laplacian_mat = spsolve(m_mat, c_mat)
        mean_curvature = 0.5 * np.linalg.norm(laplacian_mat @ self.tm.vertices, axis=1)

        # obtain M, C and Laplacian matrices
        self.lbo_m, self.lbo_c = m_mat, c_mat
        self.lbo_laplacian_mat = laplacian_mat

        return mean_curvature
    
    def compute_curvature(self, mode):
        if(mode == 'mean'):
            return self.compute_mean_curvature()

        elif(mode == 'gaussian'):
            return self.compute_gaussian_curvature()
        
        elif(mode == 'non-uniform'):
            return self.compute_non_uniform_laplace()
        

    def reconstruct(self, k):
        _ = self.compute_non_uniform_laplace()

        # Calculate M^(-1/2)
        inv_sqrt_m = la.inv(scipy.sparse.csr_matrix.sqrt(self.lbo_m))

        # Convert it into symmetric matrix 
        d_mat = inv_sqrt_m @ self.lbo_c @ inv_sqrt_m

        # Get the k-th smallest magnitude of eigenvectors (choose large magnitude with shift-invert mode)
        _, eigenvectors = eigsh(d_mat, k, which='LM', sigma=0, tol=1e-15)

        # Recover the eigenvectors with M^(-1/2)
        eigenvectors = inv_sqrt_m @ eigenvectors

        vertices = self.tm.vertices.copy()
        tmp = vertices.T @ self.lbo_m @ eigenvectors

        recon_vertices = np.zeros((self.N, 3))
        recon_vertices[:, 0] = (tmp[0] * eigenvectors).sum(axis=1)
        recon_vertices[:, 1] = (tmp[1] * eigenvectors).sum(axis=1)
        recon_vertices[:, 2] = (tmp[2] * eigenvectors).sum(axis=1)

        recon_tm = trimesh.Trimesh(vertices=recon_vertices, faces=self.tm.faces)
        return recon_tm
    
    def smooth(self, max_iters=50, lam=1e-4, mode='explicit'):
        _ = self.compute_non_uniform_laplace()

        vertices = self.tm.vertices.copy()

        assert mode in ['explicit', 'implicit'],  "mode must be 'explicit' or 'implicit'"
        if mode == 'explicit':
            for i in range(max_iters):
                vertices += lam * self.lbo_laplacian_mat @ vertices
        else:
            I = np.identity(self.N)

            a_mat = scipy.sparse.csc_matrix(I - lam * self.lbo_laplacian_mat)
            for i in range(max_iters):
                vertices = spsolve(a_mat, vertices)
        
        # update trimesh model with new vertices and faces
        smooth_tm = trimesh.Trimesh(vertices=vertices, faces=self.tm.faces)
        return smooth_tm
    
    def tm2o3d(self, tm):
        '''
        Convert trimesh object to open3d mesh
        
        models: trimesh models -> list
        '''
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(tm.vertices)
        mesh_o3d.triangles=o3d.utility.Vector3iVector(tm.faces)
        mesh_o3d.compute_vertex_normals()
        
        return mesh_o3d
    
    def show(self, curvature, cmap=cm.jet):
        '''
        Visualize the results with different colors

        Args:
        curvature: curvature of the mesh

        '''
        mesh_o3d = self.tm2o3d(self.tm)

        indices_exl_bound, _ = self.exclude_boundary_indices()

        max_curvature = np.max(curvature[indices_exl_bound])
        norm = matplotlib.colors.Normalize(vmin=np.min(curvature), vmax=max_curvature)
        print('The range of curvature: %.3f, %.3f' % (np.min(curvature), max_curvature))
        curvature = norm(curvature)

        rgb = cmap(curvature)[:, :3]
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(rgb)
        o3d.visualization.draw_geometries([mesh_o3d])

    def generate_noise(self, noise_scale):
        '''
        Generate noise on the trimesh object's vertices based on its size

        Args:
        noise_scale: the scale of amounts of noises
        '''
        vertices = self.tm.vertices.copy()

        # Find out the diagonal distance of the object
        bb_max, bb_min = np.max(vertices, axis=0), np.min(vertices, axis=0)
        bb_size = np.linalg.norm(bb_max - bb_min)

        # Add noise to the vertices
        vertices += noise_scale * bb_size * np.random.randn(vertices.shape[0], vertices.shape[1])
        self.tm.vertices = vertices
    

