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

  ListGraph::Node s = g.addNode();
  ListGraph::Node t = g.addNode();

  for(int i = 0; i < LEVELS; i++) {
	  ListGraph::Edge a = g.addEdge(s, t);
	  capacity[a] = 1;
  }

  cout << "es " << deg(g, s) << endl;

  ListGraph::NodeMap<bool> filter(g, true);
  ListGraph::EdgeMap<bool> filterE(g, true);
  filter[t] = false;
  SubGraph<ListGraph> g_(g, filter, filterE);

  GraphSubset<ListGraph> gs(g, g_);

  cout << "vol " << vol(g) << endl;
  cout << "vol sub " << vol(g_) << endl;
  cout << "vol subset " << vol(gs) << endl;
  return 0;
}

