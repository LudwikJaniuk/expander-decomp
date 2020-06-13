import math
import subprocess
import sys

assert len(sys.argv) == 2 
do_summarize = sys.argv[1] == "a" # ANalyze
out_dir = "run_output"

def source_file(graph):
    return "graphs/"+graph+".graph"

def in_partition_file(graph):
    return "partitions/5/"+graph+".2.ptn"

def out_partition_file(graph):
    return out_dir + "/" + graph+".ptn"

def output_file(graph):
    return out_dir + "/" + graph+".out"

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
    firstword = ln.split()[1]
    return int(firstword)

def run_cut_matching(graph_file, out_partition_file, print_file, g_phi, h_phi, multi, rounds, timeout, in_partition_file):
    #process = subprocess.run(["echo", graph_file, out_partition_file, str(g_phi), str(h_phi)], timeout=2)
    try:
        command = f"time -o {print_file}.time " + " ".join(["cmake-build-debug/a.out", "-f", graph_file, "-p", in_partition_file, "" if multi else "--ignore-multi",f"--timeout_m={timeout}" , "-r", f"{rounds}", "-s", f"--H_phi={h_phi}", f"--G_phi={g_phi}", "--vol", "0.1", "-o", out_partition_file, ">>", print_file])

        f = open(print_file, "w")
        f.write(command)
        f.close()
        process = subprocess.run(command, shell=True)
        print("Completed")
        print(process.returncode)
    except subprocess.TimeoutExpired:
        print("timeout")


if do_summarize:
    print("Graph_name\tvertices\tedges\tg_phi\th_phi\ttimed_out\tspent_time\tallowed_time\tread_as_multi\tCASE\tbest_cut_conductance\tbest_cut_expansion\tedges_crossing\tsize1\tsize2\tdiff_total\tdiff_div_nodes\tvol1\tvol2")

def summarize(graph_name, g_phi, h_phi, multi, timeout, graph_file, print_file, time_file):
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
    expansion = "-"
    size1 = "-"
    size2 = "-"
    diff_abs = "-"
    diff_fact = "-"
    vol1 = "-"
    vol2 = "-"
    crossing_edges = "-"
    
    # TODO consider these...
    if not timed_out:
        # Example output

# CASE2 G Expansion target reached with a cut that is relatively balanced. Cut-matching game has found a balanced cut as good as you wanted it.
# Claimed g conductance: 0.010989
# R0 cond 0.010989
# The best with best expansion was found on round0
# final_Edge crossings (E) : 1
# final_cut size: (10 | 10)
# diff: 0 (factor 0 of total n vertices)
# final_cut volumes: (91 | 91)
# final_expansion: 0.1
# final_conductance: 0.010989
# CASE2 Goodenough balanced cut

        case_line = ""
        cond_line = ""
        exp_line = ""
        vol_line = ""
        size_line = ""
        diff_line = ""
        cross_line = ""

        with open(print_file) as pf:
            lines = pf.read().splitlines()
            lines = lines[:-8] # skip partition output
            case_line = lines[-1]
            cond_line = lines[-2]
            exp_line = lines[-3]
            vol_line = lines[-4]
            diff_line = lines[-5]
            size_line = lines[-6]
            cross_line = lines[-7]

        line_int = case_line[4]

        cond_word = cond_line.split()[1]
        cond = float(cond_word)

        expansion = float(exp_line.split()[1])

        size1 = int(size_line.split()[3])
        size2 = int(size_line.split()[5])

        diff_abs = int(diff_line.split()[1])
        diff_fact = float(diff_line.split()[3])

        vol1 = int(vol_line.split()[3])
        vol2 = int(vol_line.split()[5])

        crossing_edges = int(cross_line.split()[4])

    print(f"{graph_name}\t{num_nodes_in_graph_file(graph_file)}\t{num_edges_in_graph_file(graph_file)}\t{g_phi}\t{h_phi}\t{timed_out}\t{time_used}\t{timeout}\t{multi}\t{line_int}\t{cond}\t{expansion}\t{crossing_edges}\t{size1}\t{size2}\t{diff_abs}\t{diff_fact}\t{vol1}\t{vol2}")

    #pf = open(print_file)





def run_with(graph, g_phi, h_phi, multi, rounds, timeout):
    graph_with_postfix = f"{graph}-h-{h_phi}-g-{g_phi}-t-{timeout}"

    if do_summarize:
        summarize(graph, g_phi, h_phi, multi, timeout, source_file(graph), output_file(graph_with_postfix), output_file(graph_with_postfix)+".time")
    else:
        print(f"Running on h_phi {h_phi} g_phi {g_phi} timeout {timeout}")
        run_cut_matching(source_file(graph), out_partition_file(graph_with_postfix), output_file(graph_with_postfix), g_phi, h_phi, multi, rounds, timeout, in_partition_file(graph))


def default_analyze(graph, multi):
    n_nodes = num_nodes_in_graph_file(source_file(graph))
    rounds = 0 
    g_phi = 0.0
    #g_phi = 1.0/math.log(math.log(n_nodes))
    #g_phi = 1.0/math.log(n_nodes)**2
    #g_phi = 1.0/math.sqrt(n_nodes)
    #g_phi = 1.0/n_nodes
    #for h_phi in [0.1, 0.55]:
    for h_phi in [1]: # Unreachable
        run_with(graph, g_phi, h_phi, multi, rounds, "10")

#default_analyze("barbell10-10", True)
#default_analyze("barbell100-100", True)
#default_analyze("barbell1000-1000", True)
#default_analyze("complete10", True)
#default_analyze("complete100", True)
#default_analyze("complete1000", True)
#default_analyze("expander4", True)
#default_analyze("expander16", True)
#default_analyze("expander64", True)
#default_analyze("expander256", True)
#default_analyze("expander1024", True)
#default_analyze("looploop8", True)
#default_analyze("multi8", True)

default_analyze("144", False)
default_analyze("4elt", False)
default_analyze("add32", False)
default_analyze("auto", False)
default_analyze("bcsstk33", False)
default_analyze("brack2", False)
default_analyze("fe_4elt2", False)
default_analyze("fe_sphere", False)
default_analyze("fe_tooth", False)
default_analyze("finan512", False)
default_analyze("uk", False)
default_analyze("vibrobox", False)
default_analyze("whitaker3", False)
default_analyze("wing_nodal", False)

