import subprocess

def num_nodes_in_graph_file(name):
    fo = open(name)
    ln = fo.readline()
    fo.close()
    firstword = ln.split()[0]
    return int(firstword)

def run_cut_matching(graph_file, out_partition_file, print_file, g_phi, h_phi):
    #process = subprocess.run(["echo", graph_file, out_partition_file, str(g_phi), str(h_phi)], timeout=2)
    try:
        command = f"time -o {print_file}.time timeout 15m " + " ".join(["cmake-build-debug/a.out", "-f", graph_file, "-r", "0", "-s", f"--H_phi={h_phi}", f"--G_phi={g_phi}", "--vol", "0.1", "-o", out_partition_file, ">>", print_file])

        f = open(print_file, "w")
        f.write(command)
        f.close()
        process = subprocess.run(command, shell=True)
        print("Completed")
        print(process.returncode)
    except subprocess.TimeoutExpired:
        print("timeout")

def analyze(graph):
    graph_file = "graphs/" + graph + ".graph"
    n_nodes = num_nodes_in_graph_file(graph_file)
    g_phi = 1.0/n_nodes
    print_file = "final/" + graph+".out"
    part_file = "final/" + graph+".ptn"

    print(f"Graph {graph} reading from {graph_file}, has {n_nodes} so g_phi = {g_phi}. {print_file}")

    for h_phi in [0.1, 0.4]:
        print(f"Running on h_phi {h_phi}")
        print_file = "final/" + graph+f"-hphi-{h_phi}.out"
        part_file = "final/" + graph+f"-hphi-{h_phi}.ptn"
        run_cut_matching(graph_file, part_file, print_file, g_phi, h_phi)

for line in open("final/allgraphs.txt"):
    analyze(line.split()[0])
