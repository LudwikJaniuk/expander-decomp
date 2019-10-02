#include <iostream>
#include <lemon/list_graph.h>
#include <lemon/bfs.h>
#include <lemon/dijkstra.h>
#include <lemon/preflow.h>
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

  ListDigraph g;
  ListDigraph::ArcMap<int> capacity(g);

  int LEVELS = 10000000;

  ListDigraph::Node s = g.addNode();
  ListDigraph::Node t = g.addNode();

  for(int i = 0; i < LEVELS; i++) {
	  ListDigraph::Arc a = g.addArc(s, t);
	  capacity[a] = 1;
  }
  cout << "Algo" << endl;

  Preflow<ListDigraph> preflow(g, capacity, s, t);
  preflow.run();

  cout << "Hello World! This is LEMON library here." << endl;
  cout << "We have a directed graph with " << countNodes(g) << " nodes "
       << "and " << countArcs(g) << " arc." << endl;

  cout << "maxFlow " << preflow.flowValue() << endl;
  return 0;
}

