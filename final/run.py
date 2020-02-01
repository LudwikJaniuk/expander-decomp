import subprocess

def num_nodes_in_graph_file(name):
    fo = open(name)
    ln = fo.readline()
    fo.close()
    firstword = ln.split()[0]
    return int(firstword)

def run_cut_matching(graph_file, out_partition_file, g_phi, h_phi):
    #process = subprocess.run(["echo", graph_file, out_partition_file, str(g_phi), str(h_phi)], timeout=2)
    process = subprocess.run(["cmake-build-debug/a.out", "-f", graph_file, "-r", "0", "-s", f"--H_phi={h_phi}", f"--G_phi={g_phi}", "--vol", "0.1", "-o", out_partition_filegrogro.ptn, "|", "grep" "CASE"], timeout=2)

    

def analyze(graph):
    graph_file = "graphs/" + graph + ".graph"
    n_nodes = num_nodes_in_graph_file(graph_file)
    g_phi = 1.0/n_nodes

    print(f"Graph {graph} reading from {graph_file}, has {n_nodes} so g_phi = {g_phi}")

    for h_phi in [0.1, 0.4]:
        print(f"Running on h_phi {h_phi}")
        run_cut_matching(graph_file, "final/" + graph+".ptn", g_phi, h_phi)

for line in open("final/allgraphs.txt"):
    analyze(line.split()[0])
