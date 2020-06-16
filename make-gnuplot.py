#!python3
import sys

lines = iter(sys.stdin)
next(lines)
lines = map(lambda s: s.replace("_", "\\_").split(), lines)

#0 Graph_name	vertices	edges	g_phi	h_phi	timed_out	
#6 spent_time	allowed_time	read_as_multi	CASE	best_cut_conductance	
#11 best_cut_expansion	edges_crossing	size1	size2	diff_total	
#16 diff_div_nodes	vol1	vol2	best_round	last_round
#21 walsh_cond	walsh_imb   colR    colG    colB
    

print("#vertices	conductance	our_cut_conductance	name	colors	imbalance")
#4824	0.00271739	0.00140746	"uk"	150 150 0	0.568823


for line in lines:
    print(f"{line[1]}\t{line[21]}\t{line[10]}\t{line[0]}\t{line[23]}\t{line[24]}\t{line[25]}\t{line[16]}\t")
