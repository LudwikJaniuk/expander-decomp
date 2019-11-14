#include <iostream>
#include <algorithm>
#include <random>
#include <memory>
#include <vector>
#include <array>
#include <set>
#include <lemon/adaptors.h>
#include <lemon/list_graph.h>
#include <lemon/edge_set.h>
#include <lemon/bfs.h>
#include <lemon/dijkstra.h>
#include <lemon/preflow.h>

#include "preliminaries.h"

using namespace lemon;
using namespace std;

// PARAMETERS:
int N_NODES = 1000;
int ROUNDS = 5;
bool PRINT_PATHS = false;
bool PRINT_NODES = false;
// END PARAMETERS

// Seed with a real random value, if available
random_device r;
// Choose a random mean between 1 and 6
default_random_engine engine(r());
uniform_int_distribution<int> uniform_dist(0, 1);

// Actually, cut player gets H
// Actually Actually, sure it gets H but it just needs the matchings...
template<typename G, typename M>
vector<typename G::Node> cut_player(const G& g, const vector<unique_ptr<M>>& matchings) {
	using NodeMapd = typename G::template NodeMap<double>;
	using Node = typename G::Node;
	using NodeIt = typename G::NodeIt;
	using MEdgeIt = typename M::EdgeIt;

	NodeMapd probs(g);
	vector<Node> allNodes;

	for(NodeIt n(g); n!=INVALID; ++n){
		allNodes.push_back(n);
		probs[n] = uniform_dist(engine) ? 1.0/allNodes.size() : -1.0/allNodes.size(); // TODO
	}

	if(PRINT_NODES) {
		cout << "All nodes: " << endl;
		for(const Node& n : allNodes) {
			cout << g.id(n) << " ";
		}
		cout << endl;
	}

	for(const unique_ptr<M>& m : matchings) {
		for(MEdgeIt e(*m); e!=INVALID; ++e){
			Node u = m->u(e);
			Node v = m->v(e);
			double avg = probs[u]/2 + probs[v]/2;
			probs[u] = avg;
			probs[v] = avg;
		}
	}

	sort(allNodes.begin(), allNodes.end(), [&](Node a, Node b) { 
		return probs[a] < probs[b];
	});

	size_t size = allNodes.size();
	assert(size%2==0);
	allNodes.resize(size/2);
	return allNodes;

	// for each vertex v
	// 	weight[v] = rand(0, 1) ? 1/n : -1/n
	// for each mapping m (in order)
	// 	for each edge (u,v) of m
	// 		weights[u] = weights[v] = avg(weights[u], weights[v])
	// NodeLIst nodes;
	// sort(nodelist by weight[node])
	//
	// return slice of nodellist beginning
	// (actually can optimize with stl)
	// */
}

// For some reason lemon returns arbitrary values for flow, the difference is correct tho
template<typename G>
int flow( 
	const G& g,
	const unique_ptr<Preflow<G, typename G::template EdgeMap<int>>>& f,
	typename G::Node u,
	typename G::Node v
) {
	return f->flow(findArc(g, u, v)) - f->flow(findArc(g, v, u));
}


// Very simple greedy solution
template<typename G>
inline void extract_path(
	const G& g,
	const unique_ptr<Preflow<G, typename G::template EdgeMap<int>>>& f,
	typename G::template EdgeMap<int>& subtr,
	typename G::Node u,
	typename G::Node t,
	array<typename G::Node, 2> &out_path
) {
	using Snapshot = typename G::Snapshot;
	using Node = typename G::Node;
	using Edge = typename G::Edge;
	using NodeIt = typename G::NodeIt;
	using EdgeIt = typename G::EdgeIt;
	using IncEdgeIt = typename G::IncEdgeIt;
	using OutArcIt = typename G::OutArcIt;
	using EdgeMap = typename G::template EdgeMap<int>;

	if (PRINT_PATHS) cout << "Path: ";
	out_path[0] = u;
	int prof = 0;
	for(OutArcIt a(g, u); a != INVALID; ) {
		++prof;
		Node v = g.target(a);

		int ff = flow(g, f, u, v);
		if(ff - subtr[a] <= 0) {
			++a;
			continue;
		};

		subtr[a] += 1;

		//cout << "(" << g.id(u) << " " << g.id(v) << ", " << ff <<  ")";
		if (PRINT_PATHS) cout << g.id(u);
		if(v == t) {
			out_path[1] = u;
			break;
		}
		if (PRINT_PATHS) cout << " -> ";

		u = v;
		a = OutArcIt(g, u);
	}
	if (PRINT_PATHS) cout << " ]] " << prof;
	if (PRINT_PATHS) cout << endl;

	assert(out_path.size() == 2);
}

template<typename G>
inline void extract_path_fast(
	const G& g,
	const unique_ptr<Preflow<G, typename G::template EdgeMap<int>>>& f,
	typename G::template NodeMap<vector<tuple<typename G::Node, int>>>& flow_children,
	typename G::Node u_orig,
	typename G::Node t, // For assertsions
	array<typename G::Node, 2> &out_path
) {
	using Snapshot = typename G::Snapshot;
	using Node = typename G::Node;
	using Edge = typename G::Edge;
	using NodeIt = typename G::NodeIt;
	using EdgeIt = typename G::EdgeIt;
	using IncEdgeIt = typename G::IncEdgeIt;
	using OutArcIt = typename G::OutArcIt;
	using EdgeMap = typename G::template EdgeMap<int>;

	if (PRINT_PATHS) cout << "Path: " << g.id(u_orig);
	out_path[0] = u_orig;
	Node u = u_orig;
	while(true) {
		auto& tup = flow_children[u].back();
		Node v = get<0>(tup);
		--get<1>(tup);

		if(get<1>(tup) == 0) flow_children[u].pop_back();

		if(flow_children[v].size() == 0) {
			assert(v == t);
			assert(u != u_orig);

			out_path[1] = u;
			if (PRINT_PATHS) cout <<  endl;
			break;
		}

		if (PRINT_PATHS) cout << " -> " << g.id(v);
		u = v;
	}
}

template<typename G>
vector<array<typename G::Node, 2>> decompose_paths(
	const G& g,
	const unique_ptr<Preflow<G, typename G::template EdgeMap<int>>>& f,
	typename G::Node s,
	typename G::Node t
) {
	using Snapshot = typename G::Snapshot;
	using Node = typename G::Node;
	using Edge = typename G::Edge;
	using NodeIt = typename G::NodeIt;
	using EdgeIt = typename G::EdgeIt;
	using IncEdgeIt = typename G::IncEdgeIt;
	using EdgeMap = typename G::template EdgeMap<int>;

	f->startSecondPhase();
	EdgeMap subtr(g, 0);
	vector<array<Node, 2>> paths;
	paths.reserve(N_NODES/2);

	for(IncEdgeIt e(g, s); e != INVALID; ++e) {
		Node u = g.u(e) == s ? g.v(e) : g.u(e);

		paths.push_back(array<Node, 2>());
		extract_path(g, f, subtr, u, t, paths[paths.size()-1]);
	}

	return paths;
}

template<typename G>
vector<array<typename G::Node, 2>> decompose_paths_fast(
	const G& g,
	const unique_ptr<Preflow<G, typename G::template EdgeMap<int>>>& f,
	typename G::Node s,
	typename G::Node t
) {
	using Snapshot = typename G::Snapshot;
	using Node = typename G::Node;
	using Edge = typename G::Edge;
	using NodeIt = typename G::NodeIt;
	using EdgeIt = typename G::EdgeIt;
	using IncEdgeIt = typename G::IncEdgeIt;
	using EdgeMap = typename G::template EdgeMap<int>;
	using NodeNeighborMap = typename G::template NodeMap<vector<tuple<Node, int>>>;

	f->startSecondPhase();
	EdgeMap subtr(g, 0);
	NodeNeighborMap flow_children(g, vector<tuple<Node, int>>());
	vector<array<Node, 2>> paths;
	paths.reserve(countNodes(g)/2);

	// Calc flow children (one pass)
	for(EdgeIt e(g); e != INVALID; ++e) {
		Node u = g.u(e);
		Node v = g.v(e);
		long e_flow = flow(g, f, u, v);
		if(e_flow > 0) {
			flow_children[u].push_back(tuple(v, e_flow));
		} else
		if(e_flow < 0) {
			flow_children[v].push_back(tuple(u, -e_flow));
		}
	}
	// Now path decomp is much faster

	for(IncEdgeIt e(g, s); e != INVALID; ++e) {
		assert(g.u(e) == s || g.v(e) == s);
		Node u = g.u(e) == s ? g.v(e) : g.u(e);

		paths.push_back(array<Node, 2>());
		extract_path_fast(g, f, flow_children, u, t, paths[paths.size()-1]);
	}

	return paths;
}


// TODO acutally spit out mathcing
// ant then maybe also create cut, and save all?
template<typename G>
void matching_player(G& g, const set<typename G::Node>& cut, ListEdgeSet<G>& m_out) {
	using Snapshot = typename G::Snapshot;
	using Node = typename G::Node;
	using Edge = typename G::Edge;
	using NodeIt = typename G::NodeIt;
	using EdgeIt = typename G::EdgeIt;
// TODO maybe we want to go with longs
	using EdgeMap = typename G::template EdgeMap<int>;

	size_t num_verts = countNodes(g);
	assert(num_verts%2 == 0);

	Snapshot snap(g);

	Node s = g.addNode();
	Node t = g.addNode();
	EdgeMap capacity(g);
	int s_added = 0; 
	int t_added = 0; 
	for(NodeIt n(g); n != INVALID; ++n) {
		if (n == s) continue;
		if (n == t) continue;
		Edge e;
		if(cut.count(n)) {
			e = g.addEdge(s, n);
			s_added++;
		} else {
			e = g.addEdge(n, t);
			t_added++;
		}
		capacity[e] = 1;
	}
	assert(s_added == t_added);


	unique_ptr<Preflow<G, EdgeMap>> p(new Preflow<G, EdgeMap>(g, capacity, s, t));
	for(unsigned long long i = 1; i < num_verts; i *= 2) { 

		for(EdgeIt e(g); e != INVALID; ++e) {
			if(g.u(e) == s || g.v(e) == s) continue;
			if(g.u(e) == t || g.v(e) == t) continue;
			capacity[e] = i;
		}

		p.reset(new Preflow<G, EdgeMap>(g, capacity, s, t));
		p->runMinCut();
		// Note that "startSecondPhase" must be run to get flows for individual verts
		cout << "(cap, flow): (" << i << ", " << p->flowValue() << ")" << endl;
		if(p->flowValue() == num_verts/2) {
			cout << "We have achieved full flow, but half this capacity didn't manage that!" << endl;
			break;
		}
	}

	cout << "Decomposing paths." << endl;
	auto paths = decompose_paths_fast(g, p, s, t);

	snap.restore();

	for(auto &path : paths) {
		m_out.addEdge(path[0], path.back());
	}

	// Set up source and sink to both sides
	// give all internal ones capacity of 1
	// If we find a flow of n/2
	// 	then we're done, since the cut player couldn't find a good cut,
	// 	G is already an expander...
	// 	However we're supposed to output a certification in that case right? 
	// 	And H isn't necessarily an expander yet (how will we know?) TODO
	// 	But for now if this is the case, we output claim "G is expander".
	// If not, then we double all the capacities and see if we can do it now
	// When you finally can do it and have a flow F
	// Decompose F into flow paths
	// and output those matchings. 
	//
	// I think we can do decomposition by simply going vertex by vertex and tracing their flows,
	// then subtracting that from the total flow...
	
}

void generate_large(ListGraph& g) {
	vector<ListGraph::Node> nodes;
	for(int i = 0; i < N_NODES; i++) {
		nodes.push_back(g.addNode());
	}

	g.addEdge(nodes[0], nodes[1]);
	g.addEdge(nodes[1], nodes[2]);
	g.addEdge(nodes[2], nodes[0]);

	int lim1 = N_NODES/3;
	int lim2 = 2*N_NODES/3;

	for(int i = 3; i < lim1; i++) {
		ListGraph::Node u = nodes[i];
		ListGraph::Node v = nodes[0];
		g.addEdge(u, v);
	}
	for(int i = lim1; i < lim2; i++) {
		ListGraph::Node u = nodes[i];
		ListGraph::Node v = nodes[1];
		g.addEdge(u, v);
	}
	for(int i = lim2; i < N_NODES; i++) {
		ListGraph::Node u = nodes[i];
		ListGraph::Node v = nodes[2];
		g.addEdge(u, v);
	}
}


void generate_small(ListGraph& g) {
	vector<ListGraph::Node> nodes;
	for(int i = 0; i < 10; i++) {
		nodes.push_back(g.addNode());
	}

	g.addEdge(nodes[0], nodes[1]);

	g.addEdge(nodes[0], nodes[2]);
	g.addEdge(nodes[0], nodes[3]);
	g.addEdge(nodes[0], nodes[4]);
	g.addEdge(nodes[0], nodes[5]);
	g.addEdge(nodes[1], nodes[6]);
	g.addEdge(nodes[1], nodes[7]);
	g.addEdge(nodes[1], nodes[8]);
	g.addEdge(nodes[1], nodes[9]);
}

void generate_graph(ListGraph& g) {
	generate_large(g);
}

void run() {
	ListGraph g;
	generate_graph(g);

	// Matchings
	vector<unique_ptr<ListEdgeSet<ListGraph>>> matchings;

	for(int i = 0; i < ROUNDS; i++) {
		vector<ListGraph::Node> out = cut_player<ListGraph>(g, matchings);
		if(PRINT_NODES) {
			cout << "Cut player gave the following cut: " << endl;
			for(ListGraph::Node n : out) {
				cout << g.id(n) << ", ";
			}
			cout << endl;
		}

		unique_ptr<ListEdgeSet<ListGraph>> m(new ListEdgeSet<ListGraph>(g));
		set<ListGraph::Node> cut(out.begin(), out.end());
		matching_player<ListGraph>(g, cut, *m);
		if(PRINT_NODES) {
			cout << "Matching player gave the following matching: " << endl;
			for(ListEdgeSet<ListGraph>::EdgeIt e(*m); e != INVALID; ++e) {
				cout << "(" << m->id(m->u(e)) << ", " << m->id(m->v(e)) << "), ";
			}
			cout << endl;
		}

		matchings.push_back(move(m));
		cout << "======================" << endl;
		cout << "== End round " << i << endl;
		cout << "======================" << endl;
	}
}

int main(int  argc, char** argv)
{
	if(argc >= 2) {
		N_NODES = stoi(argv[1]);
	}
	if(argc >= 3) {
		ROUNDS = stoi(argv[2]);
	}
	if(argc >= 4) {
		PRINT_PATHS = stoi(argv[3]);
	}
	if(argc >= 5) {
		PRINT_NODES = stoi(argv[4]);
	}

	run();

	return 0;
}


