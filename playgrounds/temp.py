import numpy as np
#
# k = 5  # Adjust as needed
# n = 10  # Adjust as needed
#
# # Generate random data for demonstration
# data = [np.random.rand(n) for _ in range(k)]
#
# # Initialize a tensor to store the sum of squared differences
# sum_squared_diffs = np.zeros([n] * k)
#
# # Compute squared differences for each unique pair and accumulate the results
# for i in range(k):
#     for j in range(i + 1, k):
#         # Reshape the arrays for broadcasting
#         shape_i = [n if dim == i else 1 for dim in range(k)]
#         shape_j = [n if dim == j else 1 for dim in range(k)]
#         data_i_expanded = data[i].reshape(shape_i)
#         data_j_expanded = data[j].reshape(shape_j)
#
#         # Compute the squared differences and sum across all dimensions
#         squared_diffs = (data_i_expanded - data_j_expanded) ** 2
#         sum_squared_diffs += squared_diffs

import numpy as np

# Example parameters
n = 10  # number of samples
d = 3   # dimension of vectors
k = 5   # number of variables

# Generate random vector data for demonstration
data = np.random.rand(n, d, k)

# Initialize a tensor to store the sum of norms of vector differences
sum_norms_diffs = np.zeros([n] * k)

# Compute norms of vector differences for each unique pair and accumulate the results
for i in range(k):
    for j in range(i + 1, k):
        # Extract vectors for variables i and j
        vectors_i = data[:, :, i][:, np.newaxis, :]
        vectors_j = data[:, :, j][np.newaxis, :, :]

        # Compute the norm of the vector differences
        vector_diffs = vectors_i - vectors_j
        norms = np.linalg.norm(vector_diffs, axis=2)  # Compute norms along the vector dimension

        # Prepare to broadcast norms into the tensor
        # Create an array of 1s with length k for reshaping
        broadcast_shape = [1] * k
        broadcast_shape[i] = n
        broadcast_shape[j] = n
        norms_reshaped = norms.reshape(broadcast_shape)

        # Sum the broadcasted norms into the tensor
        sum_norms_diffs += norms_reshaped

# sum_norms_diffs now contains the required sum of norms of vector differences
print("Tensor shape:", sum_norms_diffs.shape)
