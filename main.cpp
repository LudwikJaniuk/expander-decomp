#include <iostream>
#include <algorithm>
#include <random>
#include <memory>
#include <vector>
#include <array>
#include <set>
#include <chrono> 
#include <cstdlib> 
#include <bits/stdc++.h>
#include <lemon/core.h>
#include <lemon/adaptors.h>
#include <lemon/list_graph.h>
#include <lemon/edge_set.h>
#include <lemon/preflow.h>

#include "cxxopts.hpp"
#include "preliminaries.h"

using namespace lemon;
using namespace std;
using namespace std::chrono; 

// PARAMETERS:
int N_NODES = 1000;
int N_ROUNDS = 5;
bool PRINT_PATHS = false;
bool PRINT_NODES = false;
bool READ_GRAPH_FROM_FILE = false;
string IN_GRAPH_FILE;
// END PARAMETERS

// Choose a random mean between 1 and 6
default_random_engine engine;
uniform_int_distribution<int> uniform_dist(0, 1);

template<class G>
struct CutMatching {
    using NodeMapd = typename G::template NodeMap<double>;
    using Node = typename G::Node;
    using NodeIt = typename G::NodeIt;
    using Snapshot = typename G::Snapshot;
    using Edge = typename G::Edge;
    using EdgeIt = typename G::EdgeIt;
    using IncEdgeIt = typename G::IncEdgeIt;
    using OutArcIt = typename G::OutArcIt;
    using ArcLookup = ArcLookUp<G>;
    // LEMON uses ints internally. We might want to look into this
    template<class T>
    using EdgeMap = typename G::template EdgeMap<T>;
    using EdgeMapi = EdgeMap<int>;
    template<class T>
    using NodeMap = typename G::template NodeMap<T>;
    using NodeNeighborMap = NodeMap<vector<tuple<Node, int>>>;

    // Actually, cut player gets H
// Actually Actually, sure it gets H but it just needs the matchings...
    template<typename M>
    vector<Node> cut_player(const G &g, const vector<unique_ptr<M>> &matchings) {
        using MEdgeIt = typename M::EdgeIt;


        NodeMapd probs(g);
        vector<Node> allNodes;

        for (NodeIt n(g); n != INVALID; ++n) {
            allNodes.push_back(n);
            probs[n] = uniform_dist(engine) ? 1.0 / allNodes.size() : -1.0 / allNodes.size(); // TODO
        }

        if (PRINT_NODES) {
            cout << "All nodes: " << endl;
            for (const Node &n : allNodes) {
                cout << g.id(n) << " ";
            }
            cout << endl;
        }

        for (const unique_ptr<M> &m : matchings) {
            for (MEdgeIt e(*m); e != INVALID; ++e) {
                Node u = m->u(e);
                Node v = m->v(e);
                double avg = probs[u] / 2 + probs[v] / 2;
                probs[u] = avg;
                probs[v] = avg;
            }
        }

        sort(allNodes.begin(), allNodes.end(), [&](Node a, Node b) {
            return probs[a] < probs[b];
        });

        size_t size = allNodes.size();
        assert(size % 2 == 0);
        allNodes.resize(size / 2);
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
    inline
    int flow(
            const ArcLookUp<G> &alp,
            const unique_ptr<Preflow<G, EdgeMapi>> &f,
            Node u,
            Node v
    ) {
        return f->flow(alp(u, v)) - f->flow(alp(v, u));
    }

    inline void extract_path_fast(
            const G &g,
            const unique_ptr<Preflow<G, EdgeMapi>> &f,
            NodeNeighborMap &flow_children,
            Node u_orig,
            Node t, // For assertsions
            array<Node, 2> &out_path
    ) {
        if (PRINT_PATHS) cout << "Path: " << g.id(u_orig);
        out_path[0] = u_orig;
        Node u = u_orig;
        while (true) {
            auto &tup = flow_children[u].back();
            Node v = get<0>(tup);
            --get<1>(tup);

            if (get<1>(tup) == 0) flow_children[u].pop_back();

            if (flow_children[v].empty()) {
                assert(v == t);
                assert(u != u_orig);

                out_path[1] = u;
                if (PRINT_PATHS) cout << endl;
                break;
            }

            if (PRINT_PATHS) cout << " -> " << g.id(v);
            u = v;
        }
    }

    vector<array<Node, 2>> decompose_paths_fast(
            const G &g,
            const unique_ptr<Preflow<G, EdgeMapi>> &f,
            Node s,
            Node t
    ) {
        cout << "Starting to decompose paths" << endl;
        f->startSecondPhase();
        EdgeMapi subtr(g, 0);
        NodeNeighborMap flow_children(g, vector<tuple<Node, int>>());
        vector<array<Node, 2>> paths;
        paths.reserve(countNodes(g) / 2);

        cout << "Starting to pre-calc flow children" << endl;
        auto start = high_resolution_clock::now();
        // Calc flow children (one pass)
        ArcLookup alp(g);
        for (EdgeIt e(g); e != INVALID; ++e) {
            Node u = g.u(e);
            Node v = g.v(e);
            long e_flow = flow(alp, f, u, v);
            if (e_flow > 0) {
                flow_children[u].push_back(tuple(v, e_flow));
            } else if (e_flow < 0) {
                flow_children[v].push_back(tuple(u, -e_flow));
            }
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Pre-calculated path children in (microsec) " << duration.count() << endl;
        // Now path decomp is much faster

        for (IncEdgeIt e(g, s); e != INVALID; ++e) {
            assert(g.u(e) == s || g.v(e) == s);
            Node u = g.u(e) == s ? g.v(e) : g.u(e);

            paths.push_back(array<Node, 2>());
            extract_path_fast(g, f, flow_children, u, t, paths[paths.size() - 1]);
        }

        return paths;
    }


// TODO acutally spit out mathcing
// ant then maybe also create cut, and save all?
    void matching_player(G &g, const set<Node> &cut, ListEdgeSet<G> &m_out) {

        size_t num_verts = countNodes(g);
        assert(num_verts % 2 == 0);

        Snapshot snap(g);

        Node s = g.addNode();
        Node t = g.addNode();
        EdgeMapi capacity(g);

        int s_added = 0;
        int t_added = 0;
        for (NodeIt n(g); n != INVALID; ++n) {
            if (n == s) continue;
            if (n == t) continue;
            Edge e;
            if (cut.count(n)) {
                e = g.addEdge(s, n);
                s_added++;
            } else {
                e = g.addEdge(n, t);
                t_added++;
            }
            capacity[e] = 1;
        }

        assert(s_added == t_added);

        cout << "Running binary search on flows" << endl;
        auto start = high_resolution_clock::now();
        unique_ptr<Preflow<G, EdgeMapi>> p(new Preflow<G, EdgeMapi>(g, capacity, s, t));
        for (unsigned long long i = 1; i < num_verts; i *= 2) {

            for (EdgeIt e(g); e != INVALID; ++e) {
                if (g.u(e) == s || g.v(e) == s) continue;
                if (g.u(e) == t || g.v(e) == t) continue;
                capacity[e] = i;
            }

            p.reset(new Preflow<G, EdgeMapi>(g, capacity, s, t));

            cout << "Cap " << i << " ... " << flush;

            auto start2 = high_resolution_clock::now();
            p->runMinCut(); // Note that "startSecondPhase" must be run to get flows for individual verts
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start2);

            cout << "flow: " << p->flowValue() << " (" << duration.count() << " microsecs)" << endl;
            if (p->flowValue() == num_verts / 2) {
                cout << "We have achieved full flow, but half this capacity didn't manage that!" << endl;
                break;
            }
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Flow search took (microsec) " << duration.count() << endl;

        cout << "Decomposing paths." << endl;
        start = high_resolution_clock::now();
        auto paths = decompose_paths_fast(g, p, s, t);
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        cout << "Path decomposition took (microsec) " << duration.count() << endl;

        snap.restore();

        for (auto &path : paths) {
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

    void generate_large_graph(ListGraph &g) {
        vector<ListGraph::Node> nodes;
        nodes.reserve(N_NODES);
        for (int i = 0; i < N_NODES; i++) {
            nodes.push_back(g.addNode());
        }

        g.addEdge(nodes[0], nodes[1]);
        g.addEdge(nodes[1], nodes[2]);
        g.addEdge(nodes[2], nodes[0]);

        int lim1 = N_NODES / 3;
        int lim2 = 2 * N_NODES / 3;

        for (int i = 3; i < lim1; i++) {
            ListGraph::Node u = nodes[i];
            ListGraph::Node v = nodes[0];
            g.addEdge(u, v);
        }
        for (int i = lim1; i < lim2; i++) {
            ListGraph::Node u = nodes[i];
            ListGraph::Node v = nodes[1];
            g.addEdge(u, v);
        }
        for (int i = lim2; i < N_NODES; i++) {
            ListGraph::Node u = nodes[i];
            ListGraph::Node v = nodes[2];
            g.addEdge(u, v);
        }
    }

    void parse_chaco_format(const string& filename, ListGraph &g) {
        cout << "Reading graph from " << filename << endl;
        ifstream file;
        file.open(filename);
        if (!file) {
            cerr << "Unable to read file " << filename << endl;
            exit(1);
        }

        string line;
        stringstream ss;
        getline(file, line);
        ss.str(line);

        int n_verts, n_edges;
        ss >> n_verts >> n_edges;
        cout << "Reading a graph with V " << n_verts << "E " << n_edges << endl;
        g.reserveNode(n_verts);
        g.reserveNode(n_edges);

        vector<ListGraph::Node> nodes;
        for (size_t i = 0; i < n_verts; i++) {
            nodes.push_back(g.addNode());
        }

        for (size_t i = 0; i < n_verts; i++) {
            getline(file, line);
            ss.clear();
            ss << line;
            Node u = nodes[i];
            size_t v_name;
            while (ss >> v_name) {
                Node v = nodes[v_name - 1];
                if (findEdge(g, u, v) == INVALID) {
                    g.addEdge(u, v);
                }
            }
        }

        if (n_verts % 2 != 0) {
            cout << "Odd number of vertices, adding extra one." << endl;
            g.addEdge(nodes[0], g.addNode());
        }
    }

    void create_graph(ListGraph &g) {
        if (READ_GRAPH_FROM_FILE)
            parse_chaco_format(IN_GRAPH_FILE, g);
        else {
            ListGraph &g1 = g;
            cout << "Generating graph with " << N_NODES << " nodes." << endl;
            generate_large_graph(g1);
        }
    }

    void one_round(ListGraph &g, vector<unique_ptr<ListEdgeSet<ListGraph>>> &matchings) {
        cout << "Running Cut player" << endl;
        vector<ListGraph::Node> out = cut_player(g, matchings);
        if (PRINT_NODES) {
            cout << "Cut player gave the following cut: " << endl;
            for (ListGraph::Node n : out) {
                cout << ListGraph::id(n) << ", ";
            }
            cout << endl;
        }

        unique_ptr<ListEdgeSet<ListGraph>> m(new ListEdgeSet<ListGraph>(g));
        set<ListGraph::Node> cut(out.begin(), out.end());
        cout << "Running Matching player" << endl;
        matching_player(g, cut, *m);
        if (PRINT_NODES) {
            cout << "Matching player gave the following matching: " << endl;
            for (ListEdgeSet<ListGraph>::EdgeIt e(*m); e != INVALID; ++e) {
                cout << "(" << m->id(m->u(e)) << ", " << m->id(m->v(e)) << "), ";
            }
            cout << endl;
        }

        matchings.push_back(move(m));
    }

    void run() {
        ListGraph g;
        create_graph(g);

        // Matchings
        vector<unique_ptr<ListEdgeSet<ListGraph>>> matchings;
        for (int i = 0; i < N_ROUNDS; i++) {
            one_round(g, matchings);
            cout << "======================" << endl;
            cout << "== End round " << i << endl;
            cout << "======================" << endl;
        }
    }

};

cxxopts::Options create_options() {
	cxxopts::Options options("Janiuk graph partition", "Individual project implementation if thatchapon's paper to find graph partitions. Currently only cut-matching game.");
	options.add_options()
		("f,file", "File to read graph from", cxxopts::value<std::string>()->default_value(""))
		("n,nodes", "Number of nodes in graph to generate. Should be even. Ignored if -f is set.", cxxopts::value<long>()->default_value("100"))
		("r,rounds", "Number of rounds to run cut-matching game", cxxopts::value<long>()->default_value("5"))
		("p,paths", "Whether to print paths")
		("v,verbose", "Whether to print nodes and cuts (does not include paths)")
		("s,seed", "Use a seed for RNG (optionally set seed manually)", cxxopts::value<int>()->implicit_value("1337"))
		;
	return options;
}

void parse_options(int argc, char** argv) {
	auto options = create_options();
	auto result = options.parse(argc, argv);
	if(result.count("file")) {
		READ_GRAPH_FROM_FILE = true;
		IN_GRAPH_FILE = result["file"].as<string>();

	}
	if(result.count("nodes"))
		N_NODES = result["nodes"].as<long>();
	if(result.count("rounds"))
		N_ROUNDS = result["rounds"].as<long>();
	if(result.count("verbose"))
		PRINT_NODES = result["verbose"].as<bool>();
	if(result.count("paths"))
		PRINT_PATHS = result["paths"].as<bool>();
	if(result.count("seed"))
		engine = default_random_engine(result["seed"].as<int>());
	else
		engine = default_random_engine(random_device()());

}

int main(int  argc, char** argv)
{
	parse_options(argc, argv);
	CutMatching<ListGraph> cm;
	cm.run();
	return 0;
}


