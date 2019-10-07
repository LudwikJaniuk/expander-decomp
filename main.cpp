#include <iostream>
#include <lemon/adaptors.h>
#include <lemon/list_graph.h>
#include <lemon/bfs.h>
#include <lemon/dijkstra.h>
#include <lemon/preflow.h>

#include "preliminaries.h"

using namespace lemon;
using namespace std;
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
  return 0;
}

