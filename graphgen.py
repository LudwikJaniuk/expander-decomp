import networkx as nx

def write_graph(G):
    G = nx.convert_node_labels_to_integers(G, first_label=1)
    print(len(G.nodes()), len(G.edges()))
    for line in nx.generate_adjlist(G):
        print(line.partition(' ')[2])

write_graph(nx.complete_graph(10))
            
