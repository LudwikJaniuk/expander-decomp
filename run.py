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


name_colors = {
        "uk":	"150 150 0",
        "add32":	"0 150 150",
        "bcsstk33":	"150 0 150",
        "whitaker3":	"0 150 0",
        "wing_nodal":	"150 0 0",
        "fe_4elt2":	"0 0 150",
        "vibrobox":	"100 150 0",
        "4elt":	    "150 150 100",
        "fe_sphere":	"150 100 0",
        "brack2":	"100 150 100",
        "finan512":	"150 100 100",
        "fe_tooth":	"100 150 150",
        "144":	"150 150 100",
        "auto":	"100 100 150"
        }

def colorof(name):
    if name in name_colors:
        return name_colors[name]
    return "0 0 0"

if do_summarize:
    print("Graph_name\t"
          f"vertices\t"
          f"edges\t"
          f"g_phi\t"
          f"h_phi\t"
          f"timed_out\t"
          f"spent_time\t"
          f"allowed_time\t"
          f"read_as_multi\t"
          f"CASE\t"
          f"best_cut_conductance\t"
          f"best_cut_expansion\t"
          f"edges_crossing\t"
          f"size1\t"
          f"size2\t"
          f"diff_total\t"
          f"diff_div_nodes\t"
          f"vol1\t"
          f"vol2\t"
          f"best_round\t"
          f"last_round\t"
          f"walshaw_conductance\t"
          f"walshaw_imbalance\t"
          f"colR\t"
          f"colG\t"
          f"colB\t"
          )

def deciformat(n):
    normal = f"{n:9.4}"
    if 'e' in normal:
        return f"{n:9.4f}"
    return normal

def myform(n):
    ret = deciformat(n).strip()
    if n == 0:
        return ret
    if ret[0] == '0':
        return ret[1:]
    return ret


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
    best_round = "-"
    last_round = "-"
    walsh_cond = "-"
    walsh_imb = "-"
    
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
        bestround_line = ""
        lastround_line = ""
        walsh_cond_line = ""
        walsh_imb_line = ""

        with open(print_file) as pf:
            lines = pf.read().splitlines()
            walsh_cond_line = lines[-1]
            walsh_imb_line = lines[-4]

            lines = lines[:-8] # skip partition output
            case_line = lines[-1]
            cond_line = lines[-2]
            exp_line = lines[-3]
            vol_line = lines[-4]
            diff_line = lines[-5]
            size_line = lines[-6]
            cross_line = lines[-7]
            bestround_line = lines[-8]
            lastround_line = lines[-9]

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

        best_round = int(bestround_line.split()[-1][5:])
        last_round = int(lastround_line.split()[0][1:])

        walsh_cond = float(walsh_cond_line.split()[1])
        walsh_imb = float(walsh_imb_line.split()[3])



    print(f"{graph_name}\t"
          f"{num_nodes_in_graph_file(graph_file)}\t"
          f"{num_edges_in_graph_file(graph_file)}\t"
          f"{g_phi}\t"
          f"{h_phi}\t"
          f"{timed_out}\t"
          f"{time_used}\t"
          f"{timeout}\t"
          f"{multi}\t"
          f"{line_int}\t"
          f"{myform(cond)}\t"
          f"{expansion}\t"
          f"{crossing_edges}\t"
          f"{size1}\t"
          f"{size2}\t"
          f"{diff_abs}\t"
          f"{myform(diff_fact)}\t"
          f"{vol1}\t"
          f"{vol2}\t"
          f"{best_round}\t"
          f"{last_round}\t"
          f"{myform(walsh_cond)}\t"
          f"{myform(walsh_imb)}\t"
          f"{colorof(graph_name)}\t")

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
        run_with(graph, g_phi, h_phi, multi, rounds, "120")

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

#default_analyze("144", False)
#default_analyze("4elt", False)
#default_analyze("add32", False)
#default_analyze("auto", False)
#default_analyze("bcsstk33", False)
#default_analyze("brack2", False)
#default_analyze("fe_4elt2", False)
#default_analyze("fe_sphere", False)
#efault_analyze("fe_tooth", False)
#default_analyze("finan512", False)
#default_analyze("uk", False)
#default_analyze("vibrobox", False)
#default_analyze("whitaker3", False)
#default_analyze("wing_nodal", False)

run_with("144", 0.0, 1, False, 0, "30")
run_with("4elt", 0.0, 1, False, 0, "60")
run_with("add32", 0.0, 1, False, 0, "30")
run_with("auto", 0.0, 1, False, 0, "60")
run_with("bcsstk33", 0.0, 1, False, 0, "60")
run_with("brack2", 0.0, 1, False, 0, "120")
run_with("fe_4elt2", 0.0, 1, False, 0, "60")
run_with("fe_sphere", 0.0, 1, False, 0, "120")
run_with("fe_tooth", 0.0, 1, False, 0, "120")
run_with("finan512", 0.0, 1, False, 0, "60")
run_with("uk", 0.0, 1, False, 0, "30")
run_with("vibrobox", 0.0, 1, False, 0, "30")
run_with("whitaker3", 0.0, 1, False, 0, "60")
run_with("wing_nodal", 0.0, 1, False, 0, "60")
