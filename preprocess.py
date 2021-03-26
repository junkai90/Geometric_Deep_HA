import numpy as np 
from trimesh import Trimesh
from trimesh.graph import vertex_adjacency_graph
from nibabel.freesurfer.io import read_geometry

import torch
from torch_geometric.data import Data, DataLoader


class MeshData():
    def __init__(self, files):
        if isinstance(files, tuple):
            white = read_geometry(files[0])
            pial = read_geometry(files[1])
            vertices = (pial[0] - white[0]) / 2
            faces = pial[1]
            
        else:
            mid = read_geometry(files)
            vertices = mid[0]
            faces = mid[1]

        self.vertices = vertices
        self.faces = faces
        self.mesh = Trimesh(vertices=self.vertices, faces=self.faces)
        # self.nodes = None
        # self.edges = None

    def mesh2graph(self):
        graph = vertex_adjacency_graph(self.mesh)
        # self.nodes = np.arange(len(list(graph.nodes)))
        # self.edges = list(graph.edges)
        return np.arange(len(list(graph.nodes))), list(graph.edges)

    def edge_pseudo(self):
        graph = vertex_adjacency_graph(self.mesh)
        #nodes = np.arange(len(list(graph.nodes)))

        adj = graph.adjacency()
        edges = []
        pseudo = []
        for n, nbrs in adj:
            tmp = []
            for nbr in nbrs.keys():
                edges.append([n,nbr])
                tmp.append(self.vertices[nbr]-self.vertices[n])
            weights = np.array(tmp)
            co_max = np.max(weights)
            co_min = np.min(weights)
            weights = (weights-co_min)/(co_max-co_min)
            pseudo.extend(weights.tolist())

        edges = np.array(edges)
        pseudo = np.array(pseudo)

        return edges, pseudo


class GraphData():
    def __init__(self, x, y, edge_index, pseudo):
        self.x = torch.tensor(x.T, dtype=torch.long)
        self.y = torch.tensor(y.T, dtype=torch.long)
        self.edge_index = torch.tensor(edge_index.T, dtype=torch.long)
        self.pseudo = torch.tensor(pseudo, dtype=torch.float)
        self.data = [Data(x=self.x[:,i].unsqueeze(1),
                          y=self.y[:,i].unsqueeze(1),
                          edge_index=self.edge_index, 
                          edge_attr=self.pseudo) 
                     for i in range(self.x.size[1])]

    @property
    def data(self):
        return self.data


        

