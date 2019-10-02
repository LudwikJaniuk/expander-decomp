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
  ListDigraph::Node a = g.addNode();
  ListDigraph::Node b = g.addNode();
  ListDigraph::Node c = g.addNode();
  ListDigraph::Node d = g.addNode();
  ListDigraph::Node e = g.addNode();
  ListDigraph::Node f = g.addNode();

  ListDigraph::Arc  ab = g.addArc(a, b);
  ListDigraph::Arc  ac = g.addArc(a, c);
  ListDigraph::Arc  bd = g.addArc(b, d);
  ListDigraph::Arc  ce = g.addArc(c, e);
  ListDigraph::Arc  df = g.addArc(d, f);
  ListDigraph::Arc  ef = g.addArc(e, f);

  ListDigraph::ArcMap<int> capacity(g);
  capacity[ab] = 30;
  capacity[bd] = 20;
  capacity[df] = 30;
  capacity[ac] = 20;
  capacity[ce] = 30;
  capacity[ef] = 20;

  Bfs<ListDigraph> bfs(g);
  bfs.run(a);

  Preflow<ListDigraph> preflow(g, capacity, a, f);
  preflow.run();

  cout << "Hello World! This is LEMON library here." << endl;
  cout << "We have a directed graph with " << countNodes(g) << " nodes "
       << "and " << countArcs(g) << " arc." << endl;
  int count= countOutArcs(g, a);

  cout << "Node u has degree " << count << endl;
  cout << "maxFlow " << preflow.flowValue() << endl;
  return 0;
}

