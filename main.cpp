#include <iostream>
#include <algorithm>
#include <random>
#include <memory>
#include <vector>
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

	cout << "AllNodes: " << endl;
	for(const Node& n : allNodes) {
		cout << g.id(n) << " ";
	}
	cout << endl;

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

template<typename G>
void matching_player(G& g, const set<typename G::Node>& cut) {
	using Snapshot = typename G::Snapshot;
	using Node = typename G::Node;
	using Edge = typename G::Edge;
	using NodeIt = typename G::NodeIt;
	using EdgeIt = typename G::EdgeIt;
	using EdgeMap = typename G::template EdgeMap<int>;

	size_t num_verts = countNodes(g);

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
			cout << "S" << endl;
		} else {
			e = g.addEdge(n, t);
			t_added++;
			cout << "t" << endl;
		}
		capacity[e] = 1;
	}
	cout << s_added << " S added" << endl;
	cout << t_added << " T added" << endl;
	assert(s_added == t_added);


	for(unsigned long long i = 1; i < num_verts; i *= 2) { 

		for(EdgeIt e(g); e != INVALID; ++e) {
			capacity[e] = 1;
		}

		Preflow<G, EdgeMap> p(g, capacity, s, t);
		p.runMinCut();
		cout << "FLow: " << p.flowValue() << endl;

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
	
	snap.restore();
}


void run_cut_player() {
	ListGraph g;
	vector<ListGraph::Node> nodes;
	for(int i = 0; i < 100; i++) {
		nodes.push_back(g.addNode());
	}

	g.addEdge(nodes[0], nodes[1]);
	g.addEdge(nodes[1], nodes[2]);
	g.addEdge(nodes[2], nodes[0]);

	for(int i = 3; i < 33; i++) {
		ListGraph::Node u = nodes[i];
		ListGraph::Node v = nodes[0];
		g.addEdge(u, v);
	}
	for(int i = 33; i < 66; i++) {
		ListGraph::Node u = nodes[i];
		ListGraph::Node v = nodes[1];
		g.addEdge(u, v);
	}
	for(int i = 66; i < 100; i++) {
		ListGraph::Node u = nodes[i];
		ListGraph::Node v = nodes[2];
		g.addEdge(u, v);
	}

	// Matchings
	vector<unique_ptr<ListEdgeSet<ListGraph>>> matchings;

	unique_ptr<ListEdgeSet<ListGraph>> m(new ListEdgeSet<ListGraph>(g));

	ListGraph::Node u, v;
	bool odd = true;
	for(ListGraph::NodeIt n(g); n != INVALID; ++n) {
		if(odd) {
			u = n;
		} else {
			v = n;
			m->addEdge(u, v);
			cout << "M " << g.id(u) << " " << g.id(v) << endl;
		}
		odd = !odd;
	}

	matchings.push_back(move(m));
	vector<ListGraph::Node> out = cut_player<ListGraph>(g, matchings);

	for(ListGraph::Node n : out) {
		cout << g.id(n) << ", ";
	}
	cout << endl;

	set<ListGraph::Node> cut(out.begin(), out.end());
	matching_player<ListGraph>(g, cut);
}

int main()
{

//       20
//    b ---- d
//   /30      \30
//  a          f
//   \20      /20
//    c ---- e
//       30 
//   

  ListGraph g;
  ListGraph::EdgeMap<int> capacity(g);

  int LEVELS = 100;
  int LEVELS2 = 150;

  ListGraph::Node s = g.addNode();
  ListGraph::Node t = g.addNode();
  ListGraph::Node u = g.addNode();

  for(int i = 0; i < LEVELS; i++) {
	  ListGraph::Edge a = g.addEdge(s, t);
	  capacity[a] = 1;
  }
  for(int i = 0; i < LEVELS2; i++) {
	  ListGraph::Edge a = g.addEdge(t, u);
	  capacity[a] = 1;
  }

  cout << "es " << deg(g, s) << endl;

  ListGraph::NodeMap<bool> filter(g, true);
  ListGraph::EdgeMap<bool> filterE(g, true);
  filter[t] = false;
  SubGraph<ListGraph> g_(g, filter, filterE);
  GraphSubset<ListGraph> gs(g, g_);

  ListGraph::NodeMap<bool> Tfilter(g, true);
  ListGraph::EdgeMap<bool> TfilterE(g, true);
  Tfilter[u] = false;
  SubGraph<ListGraph> Tg_(g, Tfilter, TfilterE);
  GraphSubset<ListGraph> gt(g, Tg_);

  cout << "vol " << vol(g) << endl;
  cout << "vol sub " << vol(g_) << endl;
  cout << "vol subset " << vol(gs) << endl;
  cout << "S = {s, u}, T = {s, t}" << endl;
  cout << "s t u " << g.id(s) << " " << g.id(t) << " " << g.id(u) << endl;
  cout << "E(S,T) " << n_edges_between(gs, gt) << endl;
  cout << "D(S) " << cut_size(gs) << endl;
  cout << "D(T) " << cut_size(gt) << endl;
  cout << "conductance(S) " << conductance(gs) << endl;
  cout << "conductance(T) " << conductance(gt) << endl;

  Tfilter[s] = false;

  cout << "vol " << vol(g) << endl;
  cout << "vol sub " << vol(g_) << endl;
  cout << "vol subset " << vol(gs) << endl;
  cout << "S = {s, u}, T = {s, t}" << endl;
  cout << "s t u " << g.id(s) << " " << g.id(t) << " " << g.id(u) << endl;
  cout << "E(S,T) " << n_edges_between(gs, gt) << endl;
  cout << "D(S) " << cut_size(gs) << endl;
  cout << "D(T) " << cut_size(gt) << endl;
  cout << "conductance(S) " << conductance(gs) << endl;
  cout << "conductance(T) " << conductance(gt) << endl;

  cout << "Random: " << uniform_dist(engine)<< '\n';

  // We need to run the cut player with some tests

  run_cut_player();

  return 0;
}


