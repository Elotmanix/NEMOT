from anytree import Node, RenderTree
import numpy as np
import re


def parse_tree_string(tree_string):
    """
    Parse a tree string in the form '1(2,3(4))' into an AnyTree tree structure.
    Adds index, parent_index, and is_root_flag to each node using custom attributes.
    """
    tokens = re.findall(r'\d+|\(|\)|,', tree_string)  # Tokenize the string
    node_index = 0  # To track the index of nodes

    def helper(parent_index=None):
        """ Recursive function to parse the tree string and create nodes with extra properties. """
        nonlocal node_index

        if not tokens:
            return None

        value = tokens.pop(0)
        if value.isdigit():
            node_index += 1
            # node = Node(value, matrix=np.random.rand(3, 3))  # Assign random 3x3 matrix to each node
            node = Node(value)  # Assign random 3x3 matrix to each node
            node.index = node_index-1  # Assign index to node
            node.parent_index = parent_index  # Assign parent index to node
            node.is_root_flag = (parent_index is None)  # Mark if it is the root node

            if tokens and tokens[0] == '(':
                tokens.pop(0)  # Remove '('
                while tokens[0] != ')':
                    child = helper(node.index)  # Pass the current node's index as the parent index to the child
                    if child:
                        child.parent = node  # Assign the child to the current node
                    if tokens[0] == ',':
                        tokens.pop(0)  # Remove ',' between children
                tokens.pop(0)  # Remove ')'
            return node
        return None

    root = helper()
    return root


def dfs_post_order_multiply(node):
    """ Perform post-order DFS traversal and multiply matrices from leaves to root. """
    # UPDATE FN#
    aggregated_matrix = np.identity(3)  # Start with the identity matrix

    # Process each child recursively
    for child in node.children:
        aggregated_matrix = np.dot(dfs_post_order_multiply(child), aggregated_matrix)

    # Multiply current node's matrix after processing all children
    aggregated_matrix = np.dot(node.matrix, aggregated_matrix)

    return aggregated_matrix


def generate_balanced_tree_string(k, traversal_type="pre"):
    """
    Generates a string representing a balanced binary tree of k nodes
    with labels assigned based on the specified DFS traversal type.
    Supported types: "pre", "in", "post".
    """

    class TreeNode:
        """ Helper class to represent nodes in a binary tree. """

        def __init__(self):
            self.left = None
            self.right = None
            self.label = None  # This will be filled in later

    def build_balanced_tree(n):
        """ Helper to construct a balanced binary tree with n nodes. """
        if n == 0:
            return None

        root = TreeNode()
        left_size = (n - 1) // 2
        right_size = n - 1 - left_size
        root.left = build_balanced_tree(left_size)
        root.right = build_balanced_tree(right_size)

        return root

    def assign_labels(node, traversal_type, label_generator):
        """ Assigns labels based on the traversal type. """
        if node is None:
            return

        if traversal_type == "pre":
            node.label = next(label_generator)
            assign_labels(node.left, traversal_type, label_generator)
            assign_labels(node.right, traversal_type, label_generator)

        elif traversal_type == "in":
            assign_labels(node.left, traversal_type, label_generator)
            node.label = next(label_generator)
            assign_labels(node.right, traversal_type, label_generator)

        elif traversal_type == "post":
            assign_labels(node.left, traversal_type, label_generator)
            assign_labels(node.right, traversal_type, label_generator)
            node.label = next(label_generator)

    def build_tree_string(node):
        """ Recursively build the string representation of the balanced binary tree. """
        if node is None:
            return ''

        left_string = build_tree_string(node.left)
        right_string = build_tree_string(node.right)

        if left_string or right_string:
            return f'{node.label}({left_string},{right_string})'
        else:
            return f'{node.label}'

    # Step 1: Build a balanced binary tree with k nodes
    root = build_balanced_tree(k)

    # Step 2: Assign labels according to the specified traversal type
    label_generator = iter(range(1, k + 1))  # Generator for labels 1 to k
    assign_labels(root, traversal_type, label_generator)

    # Step 3: Generate the tree string based on the labeled nodes
    tree_string = build_tree_string(root)

    return tree_string

def generate_star_tree_string(k):
    """
    Generates a string representing a star-shaped tree where node 1 is the root,
    and all other nodes are its direct children.

    Example for k=4: '1(2,3,4)'
    """
    if k == 1:
        return '1'  # If k is 1, it's just a single node tree

    children = ",".join(str(i) for i in range(2, k + 1))  # Create the children list as a comma-separated string
    return f"1({children})"

def generate_tree_string(k, tree_type, traversal_type):
    if tree_type == "star":
        string = generate_star_tree_string(k)
    elif tree_type == "bin":
        string = generate_balanced_tree_string(k, traversal_type)
    else:
        raise ValueError("Tree type must be either 'star' or 'bin'")
    return string

def create_tree(params):
    if 'tree_string' in params.keys():
        # a specific tree string is given, create a tree according to it.
        tree_string = parse_tree_string(params['tree_string'])
    else:
        tree_string = generate_tree_string(k=params.k, tree_type=params.tree_type, traversal_type=params.tree_ord)

    tree_root = parse_tree_string(tree_string)

    # print resulting tree:
    print(f'Resulting Tree:')
    for pre, fill, node in RenderTree(tree_root):
        print(f"{pre}{node.name}")

    return tree_root

