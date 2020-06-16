#!python3
import sys

lines = iter(sys.stdin)
next(lines)
lines = map(lambda s: s.replace("_", "\\_").split(), lines)

#0 Graph_name	vertices	edges	g_phi	h_phi	timed_out	
#6 spent_time	allowed_time	read_as_multi	CASE	best_cut_conductance	
#11 best_cut_expansion	edges_crossing	size1	size2	diff_total	
#16 diff_div_nodes	vol1	vol2	best_round	last_round
#21 walsh_cond	walsh_imb
print(f"{4}")

for line in lines:
    latex = (
            f"\\textbf{{{line[0]}}} & {line[1]} & \\multirow{{2}}*{{{line[19]} / {line[20]}}} &  {line[10]}   & {line[16]}   \\\\\n" 
            f" {line[7]} & {line[2]} &  &  {line[21]} & {line[22]}    \\\\ \n"
            f"\\hline"

    )
    print(latex)
