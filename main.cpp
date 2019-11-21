#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-msc32-c"
#pragma ide diagnostic ignored "cppcoreguidelines-slicing"

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
    using Paths = vector<array<Node, 2>>;
    using ArcLookup = ArcLookUp<G>;
    // LEMON uses ints internally. We might want to look into this
    template<class T>
    using EdgeMap = typename G::template EdgeMap<T>;
    using EdgeMapi = EdgeMap<int>;
    template<class T>
    using NodeMap = typename G::template NodeMap<T>;
    using NodeNeighborMap = NodeMap<vector<tuple<Node, int>>>;
    using FlowAlgo = Preflow<G, EdgeMapi>;
    using Matching = ListEdgeSet<ListGraph>;
    using Matchingp = unique_ptr<Matching>;
    using Cut = set<Node>;
    using Cutp = unique_ptr<Cut>;

    default_random_engine engine;
    uniform_int_distribution<int> uniform_dist;

    struct MatchingGraph {
        G& g;
        Node s;
        Node t;
        EdgeMapi capacity;
        size_t num_vertices;
        explicit MatchingGraph(G& g_): g(g_), capacity(g) {
            num_vertices = countNodes(g);
        }
    };



    CutMatching() : uniform_dist(0, 1) {};

    // Actually, cut player gets H
// Actually Actually, sure it gets H but it just needs the matchings...
    template<typename M>
    Cutp cut_player(const G &g, const vector<unique_ptr<M>> &matchings) {
        using MEdgeIt = typename M::EdgeIt;

        NodeMapd probs(g);
        vector<Node> all_nodes;

        for (NodeIt n(g); n != INVALID; ++n) {
            all_nodes.push_back(n);
            probs[n] = uniform_dist(engine) ? 1.0 / all_nodes.size() : -1.0 / all_nodes.size(); // TODO
        }

        if (PRINT_NODES) {
            cout << "All nodes: " << endl;
            for (const Node &n : all_nodes) {
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

        sort(all_nodes.begin(), all_nodes.end(), [&](Node a, Node b) {
            return probs[a] < probs[b];
        });

        size_t size = all_nodes.size();
        assert(size % 2 == 0);
        all_nodes.resize(size / 2);
        return Cutp(new Cut(all_nodes.begin(), all_nodes.end()));

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

    void decompose_paths_fast(const MatchingGraph &mg, const unique_ptr<FlowAlgo> &f, Paths &out_paths) {
        cout << "Starting to decompose paths" << endl;
        f->startSecondPhase();
        EdgeMapi subtr(mg.g, 0);
        NodeNeighborMap flow_children(mg.g, vector<tuple<Node, int>>());
        out_paths.reserve(countNodes(mg.g) / 2);

        cout << "Starting to pre-calc flow children" << endl;
        auto start = high_resolution_clock::now();
        // Calc flow children (one pass)
        ArcLookup alp(mg.g);
        for (EdgeIt e(mg.g); e != INVALID; ++e) {
            Node u = mg.g.u(e);
            Node v = mg.g.v(e);
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

        for (IncEdgeIt e(mg.g, mg.s); e != INVALID; ++e) {
            assert(mg.g.u(e) == mg.s || mg.g.v(e) == mg.s);
            Node u = mg.g.u(e) == mg.s ? mg.g.v(e) : mg.g.u(e);

            out_paths.push_back(array<Node, 2>());
            extract_path_fast(mg.g, f, flow_children, u, mg.t, out_paths[out_paths.size() - 1]);
        }
    }

    size_t bin_search_flows(MatchingGraph &mg, unique_ptr<FlowAlgo> &p) const {
        cout << "Running binary search on flows" << endl;
        auto start = high_resolution_clock::now();
        size_t cap = 1;
        for (; cap < mg.num_vertices; cap *= 2) {
            for (EdgeIt e(mg.g); e != INVALID; ++e) {
                if (mg.g.u(e) == mg.s || mg.g.v(e) == mg.s) continue;
                if (mg.g.u(e) == mg.t || mg.g.v(e) == mg.t) continue;
                mg.capacity[e] = cap;
            }

            p.reset(new Preflow<G, EdgeMapi>(mg.g, mg.capacity, mg.s, mg.t));

            cout << "Cap " << cap << " ... " << flush;

            auto start2 = high_resolution_clock::now();
            p->runMinCut(); // Note that "startSecondPhase" must be run to get flows for individual verts
            auto stop2 = high_resolution_clock::now();
            auto duration2 = duration_cast<microseconds>(stop2 - start2);

            cout << "flow: " << p->flowValue() << " (" << duration2.count() << " microsecs)" << endl;
            if (p->flowValue() == mg.num_vertices / 2) {
                cout << "We have achieved full flow, but half this capacity didn't manage that!" << endl;
                break;
            }
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Flow search took (microsec) " << duration.count() << endl;

        return cap;
    }

    void decompose_paths(const MatchingGraph &mg, const unique_ptr<FlowAlgo> &p, vector<array<Node, 2>> &paths) {
        cout << "Decomposing paths." << endl;
        auto start = high_resolution_clock::now();
        decompose_paths_fast(mg, p, paths);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Path decomposition took (microsec) " << duration.count() << endl;
    }

// returns capacity that was required
// TODO make the binsearch an actual binsearch
    size_t matching_player(G &g, const set<Node> &cut, ListEdgeSet<G> &m_out) {
        size_t num_verts = countNodes(g);
        assert(num_verts % 2 == 0);
        Snapshot snap(g);

        MatchingGraph mg(g);
        make_sink_source(mg, cut);

        unique_ptr<FlowAlgo> p;
        size_t cap_needed = bin_search_flows(mg, p);

        vector<array<Node, 2>> paths;
        decompose_paths(mg, p, paths);

        for (auto &path : paths) {
            m_out.addEdge(path[0], path.back());
        }
        snap.restore();

        // Now how do we extract the cut?
        // In this version, in one run of the matching the cut is strictly decided. We just need
        // to decide which one of them.
        // Only when we change to edge will the cut need to be explicitly extracted.
        // Rn the important thing is to save cuts between rounds so I can choose the best.

        return cap_needed;
    }

    void make_sink_source(MatchingGraph &mg, const set<Node> &cut) const {
        G& g = mg.g;
        mg.s = g.addNode();
        mg.t = g.addNode();
        int s_added = 0;
        int t_added = 0;
        for (NodeIt n(g); n != INVALID; ++n) {
            if (n == mg.s) continue;
            if (n == mg.t) continue;
            Edge e;
            if (cut.count(n)) {
                e = g.addEdge(mg.s, n);
                s_added++;
            } else {
                e = g.addEdge(n, mg.t);
                t_added++;
            }
            mg.capacity[e] = 1;
        }
        assert(s_added == t_added);
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

    void parse_chaco_format(const string &filename, ListGraph &g) {
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

    void generate_graph(ListGraph &g) {
        if (READ_GRAPH_FROM_FILE)
            parse_chaco_format(IN_GRAPH_FILE, g);
        else {
            ListGraph &g1 = g;
            cout << "Generating graph with " << N_NODES << " nodes." << endl;
            generate_large_graph(g1);
        }
    }

    size_t one_round(G &g, vector<Cutp> &cuts, vector<Matchingp> &matchings) {
        cout << "Running Cut player" << endl;
        Cutp cut = cut_player(g, matchings);
        if (PRINT_NODES) { print_cut(*cut); }

        Matchingp m(new Matching(g));

        cout << "Running Matching player" << endl;
        size_t cap = matching_player(g, *cut, *m);
        if (PRINT_NODES) { print_matching(m); }

        matchings.push_back(move(m));
        cuts.push_back(move(cut));
        return cap;
    }

    void print_matching(const Matchingp& m) {
        cout << "Matching player gave the following matching: " << endl;
        for (Matching::EdgeIt e(*m); e != INVALID; ++e) {
            cout << "(" << m->id(m->u(e)) << ", " << m->id(m->v(e)) << "), ";
        }
        cout << endl;
    }

    void print_cut(const Cut &out_cut) const {
        cout << "Cut player gave the following cut: " << endl;
        for (Node n : out_cut) {
            cout << G::id(n) << ", ";
        }
        cout << endl;
    }

    void run() {
        ListGraph g;
        generate_graph(g);
        size_t num_vertices = countNodes(g);

        // Matchings
        vector<Matchingp> matchings;
        vector<Cutp> cuts;

        size_t best_cap = 0;
        size_t best_cap_index = 999999;
        for (int i = 0; i < N_ROUNDS; i++) {
            size_t cap = one_round(g, cuts, matchings);
            print_end_round(i);

            if(cap > best_cap) {
                best_cap = cap;
                best_cap_index = i;
            }
        }

        const Matchingp& m = matchings[best_cap_index];
        const Cutp& cut = cuts[best_cap_index];
        cout << "The cut with highest capacity required was found on round" << best_cap_index << endl;
        cout << "Here's how sparse it was:" << endl;

        double crossing_edges = 0;
        for(EdgeIt e(g); e != INVALID; ++e) {
            if(is_crossing(g, *cut, e)) crossing_edges += 1;
        }
        assert(cut->size() <= num_vertices);
        cout << "Edge crossings (E) : " << crossing_edges << endl;
        double min_side = min(cut->size(), num_vertices - cut->size());
        double expansion_maybe = crossing_edges / min_side;

        cout << "E/min(|S|, |comp(S)|) = " << expansion_maybe << endl;
    }

    bool is_crossing(const G& g, const Cut& c, const Edge& e) {
        bool u_in = c.count(g.u(e));
        bool v_in = c.count(g.v(e));
        return u_in != v_in;
    }

    void print_end_round(int i) const {
        cout << "======================" << endl;
        cout << "== End round " << i << endl;
        cout << "======================" << endl;
    }
};

cxxopts::Options create_options() {
    cxxopts::Options options("Janiuk graph partition",
                             "Individual project implementation if thatchapon's paper to find graph partitions. Currently only cut-matching game.");
    options.add_options()
            ("f,file", "File to read graph from", cxxopts::value<std::string>()->default_value(""))
            ("n,nodes", "Number of nodes in graph to generate. Should be even. Ignored if -f is set.",
             cxxopts::value<long>()->default_value("100"))
            ("r,rounds", "Number of rounds to run cut-matching game", cxxopts::value<long>()->default_value("5"))
            ("p,paths", "Whether to print paths")
            ("v,verbose", "Whether to print nodes and cuts (does not include paths)")
            ("s,seed", "Use a seed for RNG (optionally set seed manually)",
             cxxopts::value<int>()->implicit_value("1337"));
    return options;
}

void parse_options(int argc, char **argv, CutMatching<ListGraph> &cm) {
    auto options = create_options();
    auto result = options.parse(argc, argv);
    if (result.count("file")) {
        READ_GRAPH_FROM_FILE = true;
        IN_GRAPH_FILE = result["file"].as<string>();

    }
    if (result.count("nodes"))
        N_NODES = result["nodes"].as<long>();
    if (result.count("rounds"))
        N_ROUNDS = result["rounds"].as<long>();
    if (result.count("verbose"))
        PRINT_NODES = result["verbose"].as<bool>();
    if (result.count("paths"))
        PRINT_PATHS = result["paths"].as<bool>();
    if (result.count("seed"))
        cm.engine = default_random_engine(result["seed"].as<int>());
    else
        cm.engine = default_random_engine(random_device()());

}

int main(int argc, char **argv) {
    CutMatching<ListGraph> cm;
    parse_options(argc, argv, cm);
    cm.run();
    return 0;
}

#pragma clang diagnostic pop