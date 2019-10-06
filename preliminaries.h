#include <iostream>
#include <lemon/list_graph.h>
#include <lemon/bfs.h>
#include <lemon/dijkstra.h>
#include <lemon/preflow.h>

using namespace lemon;
using namespace std;


// Like a subgraph, but keep around the orignal graph not to hide e.g. edged going out
template <typename MyGraph>
struct GraphSubset {
	MyGraph &graph;
	SubGraph<MyGraph> &subset;
	GraphSubset(MyGraph& g, SubGraph<MyGraph> &s) : graph(g), subset(s) {};
};

// 1. Notations
// Undirected graphs

template <typename MyGraph>
int deg(const MyGraph &g, const typename MyGraph::Node &n) {
	return countIncEdges(g, n);
}

// How will this work with an adaptor that hides verts?
template <typename MyGraph>
int vol(const MyGraph &g) {
	int volume = 0;
	for(typename MyGraph::NodeIt n(g); n != INVALID; ++n) {
		volume += deg(g, n);
	}
	return volume;
}

template <typename MyGraph>
int vol(const GraphSubset<MyGraph> &gs) {
	int volume = 0;
	for(typename SubGraph<MyGraph>::NodeIt n(gs.subset); n != INVALID; ++n) {
		volume += deg(gs.graph, n);
	}
	return volume;
}


