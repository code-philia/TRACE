import networkx as nx
from copy import deepcopy
from itertools import permutations, product

def dfs_with_constraints(graph):
    """
    DFS for networkx DiGraph with constraints.
    1. If a node has multiple parents, all parents must be visited before visiting the node.
    2. When there are branches, prioritize the branch that matches the current node path.
    
    Args:
        graph: NetworkX DiGraph with each node containing a "file_path" attribute
    Returns:
        List of possible traverse orders
    """
    def generate_flattened_permutations(list_of_lists):
        """
        Given a list of lists, generate all permutations of sublists and generate all permutations of elements in each sublist.
        Finally, flatten each permutation and store them in a list.

        Args:
            list_of_lists: list of list, e.g. [[1, 2], [3, 4], [5]]
        Returns:
            List of flattened permutations
        """
        # Generate the permutations of sublists
        all_permutations = permutations(list_of_lists)

        flattened_results = []

        for perm in all_permutations:
            # Generate the permutations of elements in each sublist
            sublist_permutations = [list(permutations(sublist)) for sublist in perm]

            # Generate all combinations of sublists
            all_combinations = product(*sublist_permutations)

            # Flatten each combination and store them in the final list
            for combination in all_combinations:
                flattened_results.append([item for sublist in combination for item in sublist])

        return flattened_results

    # Calculate indegree of each node
    indegree = {node: 0 for node in graph}
    for u, v in graph.edges:
        indegree[v] += 1

    undirected_graph = graph.to_undirected()
    # Init stack: all nodes with 0 indegree
    # Nodes in the same connected graph should be neighbours in stack
    connected_components = list(nx.connected_components(undirected_graph))
    grouped_root_nodes = []
    for component in connected_components:
        group_in_component = []
        for node in component:
            if indegree[node] == 0:
                group_in_component.append(node)
        grouped_root_nodes.append(group_in_component)
    
    flattened_results = generate_flattened_permutations(grouped_root_nodes)
    stack_of_stacks = []
    # Each stack contains the current stack, with current indegree of nodes and visted nodes
    for init_stack in flattened_results:
        stack_of_stacks.append({
            "stack": init_stack,
            "indegree": deepcopy(indegree),
            "result": []
        })
    
    all_results = []
    while stack_of_stacks:
        current_stack = stack_of_stacks.pop()
        
        stack = current_stack["stack"]
        result = current_stack["result"]
        indegree = current_stack["indegree"]
        
        # Find successors of the current node
        current = stack.pop()
        result.append(current)
        
        same_file_successors = []
        diff_file_successors = []
        for successor in graph.successors(current):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                if graph.nodes[successor]["file_path"] == graph.nodes[current]["file_path"]:
                    same_file_successors.append(successor)
                else:
                    diff_file_successors.insert(0, successor)
        
        permute_of_same_file_successors = permutations(same_file_successors)
        permute_of_diff_file_successors = permutations(diff_file_successors)
        
        possible_permutations = product(permute_of_same_file_successors, permute_of_diff_file_successors)
        for permute in possible_permutations:
            new_stack = stack + list(permute[1]) + list(permute[0]) # nodes in the file should be visited first, hence permute[1] first
            if new_stack:
                stack_of_stacks.append({
                    "stack": new_stack,
                    "indegree": deepcopy(indegree),
                    "result": deepcopy(result)
                })
            else:
                all_results.append(deepcopy(result))

    return all_results
