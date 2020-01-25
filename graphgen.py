import networkx as nx

def write_graph(G, f):
    G = nx.convert_node_labels_to_integers(G, first_label=1)
    f.write(str(len(G.nodes())) +" "+ str(len(G.edges()))+"\n")
    for line in nx.generate_adjlist(G):
        f.write(line.partition(' ')[2])
        f.write("\n")

#f = open("complete10.graph", "w+")
#write_graph(nx.complete_graph(10), f)
#
#f = open("complete100.graph", "w+")
#write_graph(nx.complete_graph(100), f)
#
#f = open("complete1000.graph", "w+")
#write_graph(nx.complete_graph(1000), f)

#f = open("complete10000.graph", "w+")
#write_graph(nx.complete_graph(10000), f)

#f = open("complete100000.graph", "w+")
#write_graph(nx.complete_graph(100000), f)

f = open("barbell4-1-4.graph", "w+")
write_graph(nx.barbell_graph(4, 1), f)
f = open("barbell5-5.graph", "w+")
write_graph(nx.barbell_graph(5, 0), f)
f = open("barbell6-6.graph", "w+")
write_graph(nx.barbell_graph(6, 0), f)
f = open("barbell7-7.graph", "w+")
write_graph(nx.barbell_graph(7, 0), f)
f = open("barbell8-8.graph", "w+")
write_graph(nx.barbell_graph(8, 0), f)
f = open("barbell9-9.graph", "w+")
write_graph(nx.barbell_graph(9, 0), f)
f = open("barbell10-10.graph", "w+")
write_graph(nx.barbell_graph(10, 0), f)
f = open("barbell10-1-10.graph", "w+")
write_graph(nx.barbell_graph(10, 1), f)


#f = open("barbell10-10.graph", "w+")
#write_graph(nx.barbell_graph(10, 1), f)
#
#f = open("barbell100-100.graph", "w+")
#write_graph(nx.barbell_graph(100, 1), f)
#
#f = open("barbell1000-1000.graph", "w+")
#write_graph(nx.barbell_graph(1000, 1), f)

#f = open("barbell10000-10000.graph", "w+")
#write_graph(nx.barbell_graph(10000, 10000), f)

#f = open("barbell100000-100000.graph", "w+")
#write_graph(nx.barbell_graph(100000, 100000), f)


