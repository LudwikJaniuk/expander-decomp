import re
import subprocess

def num_nodes_in_graph_file(name):
    fo = open(name)
    ln = fo.readline()
    fo.close()
    firstword = ln.split()[0]
    return int(firstword)

def num_edges_in_graph_file(name):
    fo = open(name)
    ln = fo.readline()
    fo.close()
    secondword = ln.split()[1]
    return int(secondword)

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

print("Graph_name\tvertices\tedges\tg_phi\th_phi\ttimed_out\tspent_time\tCASE\tbest_cut_conductance\tdiff_total\tdiff_div_nodes")

def summarize(graph_name, g_phi, h_phi, graph_file, print_file, time_file):
    timed_out = False
    runtime = -1

    tf = open(time_file)
    time_line = tf.readline()
    if(time_line.find("Command") == 0):
        timed_out = True
        time_line = tf.readline()
    tf.close()
    user_index = time_line.find("user")
    time_string = time_line[:user_index]
    time_used = float(time_string)

    line_int = "-"
    cond = "-"
    diff_abs = "-"
    diff_fact = "-"
    if not timed_out:
        case_line = ""
        cond_line = ""
        diff_line = ""
        with open(print_file) as pf:
            lines = pf.read().splitlines()
            case_line = lines[-1]
            cond_line = lines[-4]
            diff_line = lines[-7]
        line_int = case_line[4]
        cond_word = cond_line.split()[1]
        cond = float(cond_word)

        diff_abs = int(diff_line.split()[1])
        diff_fact = float(diff_line.split()[2][1:])


    print(f"{graph_name}\t{num_nodes_in_graph_file(graph_file)}\t{num_edges_in_graph_file(graph_file)}\t{g_phi}\t{h_phi}\t{timed_out}\t{time_used}\t{line_int}\t{cond}\t{diff_abs}\t{diff_fact}")

    #pf = open(print_file)




def analyze(graph):
    graph_file = "graphs/" + graph + ".graph"
    n_nodes = num_nodes_in_graph_file(graph_file)
    g_phi = 1.0/n_nodes
    print_file = "final/" + graph+".out"
    part_file = "final/" + graph+".ptn"

    for h_phi in [0.1, 0.55]:
        print_file = "final/" + graph+f"-hphi-{h_phi}.out"
        part_file = "final/" + graph+f"-hphi-{h_phi}.ptn"
        #run_cut_matching(graph_file, part_file, print_file, g_phi, h_phi)
        summarize(graph, g_phi, h_phi, graph_file, print_file, print_file+".time")

for line in open("final/allgraphs.txt"):
    analyze(line.split()[0])
