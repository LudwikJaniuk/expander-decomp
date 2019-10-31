#include <iostream>
#include <algorithm>
#include <random>
#include <lemon/adaptors.h>
#include <lemon/list_graph.h>
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

  return 0;
}

template<typename G>
vector<int> cut_player(const G& g, const vector<G>& matchings) {
	using NodeMapd = typename G::template NodeMap<double>;
	using Node = typename G::Node;
	using NodeIt = typename G::NodeIt;
	using EdgeIt = typename G::EdgeIt;

	NodeMapd probs;
	vector<Node> allNodes(countNodes(g));
	return allNodes;

	for(NodeIt n(g); n!=INVALID; ++n){
		allNodes.push_back(n);
		probs[n] = uniform_dist(engine) ? 1.0/allNodes.size() : -1.0/allNodes.size(); // TODO
	}

	for(const G& m : matchings) {
		for(EdgeIt e(g); e!=INVALID; ++e){
			Node u = m.u(e);
			Node v = m.v(e);
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


