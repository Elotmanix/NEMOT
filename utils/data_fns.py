import os
import ssl
import numpy as np  # Ensure NumPy is imported
import torch
from torch.utils.data import Dataset, DataLoader
# import jax.numpy as jnp
# import jax.random as jr
from torchvision import datasets, transforms
from PIL import Image
import torchvision.transforms.functional as TF



def gen_data(params, dataset=None):
    if params.data_dist == 'uniform':
        if params['euler'] == 1:
            '''
            generate data on a sphere
            '''
            X = np.random.randn(params['n'], params['dim'] + 1)
            X /= np.linalg.norm(X, axis=-1)[:, np.newaxis]
            X = X.T
            X = np.array([X] * params['k'])
            X = X.transpose(2, 1, 0)

        else:
            X = [np.random.rand(params['n'], params['dim']) for i in range(params['k'])]
            X = np.array(X)
            X = X.transpose(1, 2, 0)

        if params['alg'] == 'sinkhorn_mot':
            n = X.shape[0]
            MU = [(1 / n) * np.ones(n)] * params['k']
            return X, MU
        return X

    elif params.data_dist == 'gaussian':
        X = [np.random.randn(params['n'], params['dim']) * params['gauss_std'] for i in range(params['k'])]
        X = np.array(X)
        X = X.transpose(1, 2, 0)
        if params['alg'] == 'sinkhorn_mot':
            n = X.shape[0]
            MU = [(1 / n) * np.ones(n)] * params['k']
            return X, MU
        return X

    elif params.dataset == 'mnist':
        X = gen_mnist_tensor(params)
        if params['alg'] == 'sinkhorn_mot':
            n = X.shape[0]
            MU = [(1 /n) * np.ones(n)]*params['k']
            return X, MU
        return X

    else:
        raise ValueError(f"Unsupported data distribution: {params.data_dist} or dataset: {params.dataset}")
            MU = [(1 / params['n']) * np.ones(params['n'])]*params['k']
            return X, MU
        elif params['alg'] == 'ne_mot':
            X = torch.from_numpy(np.stack(X, axis=-1))
            X = [torch.from_numpy(x) for x in X]
            return X


def CoulombCost(data):
    '''
    Calculate the Coulomb cost: sum_{i<j} 1/||r_i - r_j||
    Args:
        data: tensor of shape (batch_size, d, k) containing k d-dimensional points
              where d must be 1, 2, or 3 (physical space dimension)
    Returns:
        tensor of shape (batch_size,) containing the Coulomb cost for each batch
    '''
    k = data.shape[2]
    d = data.shape[1]
    if d not in [1, 2, 3]:
        raise ValueError(f"Spatial dimension must be 1, 2, or 3, got {d}")
    
    costs = []
    eps = 1e-6  # Small constant to avoid division by zero
    
    for i in range(k):
        for j in range(i+1, k):  # Note: j>i to avoid double counting
            diff = data[:,:,i].unsqueeze(2) - data[:,:,j].unsqueeze(1)
            # Calculate Euclidean distance (adding small eps to avoid division by zero)
            dist = torch.sqrt(torch.sum(diff**2, dim=-1) + eps)
            # Calculate 1/r Coulomb potential
            costs.append(1.0 / dist)
    
    costs = torch.stack(costs)
    return costs


def QuadCostGW(data, matrices, mod='circle'):
    k = data.shape[-1]
    c = []
    for i in range(k):
        c.append(data[:, :, i]@matrices[i]@data[:,:,(i+1)%k].T)
    return c

    """
    Calculate the (k-1) squared L2 distance matrices for each non-root node in the tree.

    Parameters:
    - X: torch.Tensor of shape (n, d, k), where n is the number of samples, d is the dimension of each vector, and k is the number of nodes.
    - root: The root of the tree, which is a Node object with children.

    Returns:
    - A list of torch.Tensor matrices, each of shape (n, n), representing the squared L2 distance for each non-root node.
    """
    n, d, k = X.shape
    matrices = [0]*k

    def traverse_and_calculate(node):
        # If the node is not the root, calculate the squared L2 distance matrix
        if not node.is_root_flag:
            parent_index = node.parent_index   # Adjust for zero-indexing in Python
            node_index = node.index   # Adjust for zero-indexing in Python

            # Efficient broadcasting-based computation for squared L2 norms
            # # Implementation where C[i,j]=x[i]-parent[j]
            # vectors_i = X[:, :, node_index].unsqueeze(1)  # (n, 1, d)
            # vectors_j = X[:, :, parent_index].unsqueeze(0)  # (1, n, d)

            # Implementation where C[i,j]=parent[i]-x[j]
            vectors_i = X[:, :, node_index].unsqueeze(0)  # (n, 1, d)
            vectors_j = X[:, :, parent_index].unsqueeze(1)  # (1, n, d)

            # Compute the squared L2 distance between all pairs using broadcasting
            vector_diffs = vectors_i - vectors_j  # (n, n, d)
            C_i = torch.norm(vector_diffs, dim=2) ** 2  # (n, n), squared L2 norms

            # Store the matrix for this node
            matrices[node_index] = C_i

        # Traverse to the children recursively
        for child in node.children:
            traverse_and_calculate(child)

    # Traverse the tree starting from the root and calculate the matrices
    traverse_and_calculate(root)

    # CURRENTLY FOR ROOT AT idx=0
    return matrices

def kronecker_product(vectors):
    for index in range(1, len(vectors)):
        if index == 1:
            out = np.tensordot(vectors[index - 1], vectors[index], axes=0)
        else:
            out = np.tensordot(out, vectors[index], axes=0)
    return out

def calc_ent(p):
    return -np.sum(p*np.log(p))


def EulerSigma(data, case=0):
    """
    Applies the Euler Flow sigma displacement function.
    :param data: assumed to be one-dimensional, so shape is (n,d,1)
    :return:
    """
    if case == 0:
        # in this case its x = (x + 0.5) mod 1
        data = (data + 0.5)%1
        return data


def style_transfer_data(params):
    """
    Creating the data for style transfer experiment.
    1. Loading images
    2. Loading vgg
    3. mapping images into data
    4. returning encoded images and decoder models
    """


class MultiTensorDataset(Dataset):
    def __init__(self, tensor_list):
        """
        Initialize the dataset with a list of tensors.
        :param tensor_list: List of tensors with the same number of samples (first dimension).
        """
        self.tensors = tensor_list
        self.n_samples = tensor_list[0].shape[0]  # Number of samples
        # Ensure all tensors have the same number of samples
        assert all(tensor.shape[0] == self.n_samples for tensor in self.tensors), \
            "All tensors must have the same number of samples (first dimension)."

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        """
        Return the sample at the given index as a tuple of tensors.
        """
        return tuple(tensor[index] for tensor in self.tensors)


def gen_mnist(params):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize to mean 0.1307 and std 0.3081
        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))  # Map back to [0, 1]
    ])
    mnist_path = './data/mnist_data/'
    if not os.path.exists(mnist_path):
        raise RuntimeError(f"MNIST dataset not found at {mnist_path}")

    # Disable SSL verification
    ssl._create_default_https_context = ssl._create_unverified_context
    
    mnist_train = datasets.MNIST(root=mnist_path, train=False, download=True, transform=transform)
    mnist_loader = DataLoader(mnist_train, batch_size=params['batch_size'], shuffle=True)
    return mnist_loader

def gen_mnist_tensor(params):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize to mean 0.1307 and std 0.3081
        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))  # Map back to [0, 1]
    ])
    mnist_path = './data/mnist_data/'
    if not os.path.exists(mnist_path):
        raise RuntimeError(f"MNIST dataset not found at {mnist_path}")

    # Disable SSL verification
    ssl._create_default_https_context = ssl._create_unverified_context
    
    mnist_train = datasets.MNIST(root=mnist_path, train=False, download=True, transform=transform)
    data = mnist_train.data.float().view(-1, 784)
    labels = mnist_train.targets
    class_data = {i: [] for i in range(10)}

    for img, label in zip(data, labels): # collect data according to labels:
        class_data[int(label)].append(img)
    for i in range(10): # stack into tensors:
        class_data[i] = torch.stack(class_data[i], dim=0)
    n = min(class_data[i].shape[0] for i in range(10))

    balanced_data = []
    for i in range(10):
        balanced_data.append(class_data[i][:n])

    mnist_tensor = torch.stack(balanced_data, dim=-1)

    mnist_tensor = mnist_tensor[:,:,:params.k]

    return mnist_tensor

def gen_data_JAX(params):
    import jax.numpy as jnp
    import jax.random as jr
    key = jr.PRNGKey(params.get('seed', 0))  # Use a seed for reproducibility, default to 0

    if params.dataset == 'mnist':
        X = gen_mnist(params)
    elif params['data_dist'] == 'uniform':
        if params.get('euler', 0) == 1:  # Use .get() for safety, default to 0
            X = [jnp.linspace(0, 1, params['n'], dtype=jnp.float32).reshape(params['n'], 1) for _ in range(params['k'])]
        else:
            X = []
            for i in range(params['k']):
                key, subkey = jr.split(key)
                low = -1 / jnp.sqrt(params['dims'][i])
                high = 1 / jnp.sqrt(params['dims'][i])
                x = jr.uniform(subkey, (params['n'], params['dims'][i]), minval=low, maxval=high, dtype=jnp.float32)
                X.append(x)

    elif params['data_dist'] == 'gauss':
        X = []
        for i in range(params['k']):
            key, subkey = jr.split(key)
            std = params['gauss_std'] / jnp.sqrt(params['dims'][i])
            x = std * jr.normal(subkey, (params['n'], params['dims'][i]), dtype=jnp.float32)
            X.append(x)
    
    else:
        raise ValueError(f"Unknown data distribution: {params['data_dist']}")
    if params.dataset != 'mnist':
        if params['alg'] != 'ne_mot':
            X = jnp.stack(X, axis=-1)
            return X
        
    return X


def rotate(img, angle=15):
    """
    Rotate the image by the specified angle (in degrees).
    
    Args:
        angle: Rotation angle in degrees (default is 15) - don't exceed (-15,15).
    
    Returns:
        Rotated image.
    """
    return TF.rotate(img, angle)


def translate(img, translate_vector=(2, 0)):
    """
    Translate the image using an affine transformation.
    
    Args:
        translate_vector: A tuple (tx, ty) indicating pixel shifts (default is (2, 0)) - don exceed (+-2,+-2).
    
    Returns:
        Translated image.
    """
    # angle is 0 and scale is 1.0, shear is 0 by default for a pure translation
    return TF.affine(img, angle=0, translate=translate_vector, scale=1.0, shear=0)


def perspective_warp(img, 
                     startpoints=[(0, 0), (28, 0), (0, 28), (28, 28)], 
                     endpoints=[(2, 2), (26, 0), (2, 26), (28, 28)]):
    """
    Apply a perspective warp to the image based on provided points.
    
    Args:
        startpoints: List of four tuples indicating the source coordinates.
        endpoints: List of four tuples indicating the destination coordinates.
        
    Returns:
        Warped image.
    """
    return TF.perspective(img, startpoints, endpoints)