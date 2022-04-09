import os
from typing import Dict, Set
from exercises.tick10 import load_graph


def get_node_betweenness(graph: Dict[int, Set[int]]) -> Dict[int, float]:
    """
    Use Brandes' algorithm to calculate the betweenness centrality for each node in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: a dictionary mapping each node ID to that node's betweenness centrality
    """
    pass


def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'simple_network.edges'))

    betweenness = get_node_betweenness(graph)
    print(f"Node betweenness values: {betweenness}")


if __name__ == '__main__':
    main()
