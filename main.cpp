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
#include <lemon/connectivity.h>
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
bool COMPARE_PARTITION = false;
string PARTITION_FILE;
bool OUTPUT_CUT = false;
string OUTPUT_FILE;
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
    using NodeMapi = NodeMap<int>;
    using NodeNeighborMap = NodeMap<vector<tuple<Node, int>>>;
    using FlowAlgo = Preflow<G, EdgeMapi>;
    using Matching = ListEdgeSet<ListGraph>;
    using Matchingp = unique_ptr<Matching>;
    using Bisection = set<Node>;
    using Bisectionp = unique_ptr<Bisection>;
    using Cut = set<Node>;
    using Cutp = unique_ptr<Cut>;
    using CutMap = NodeMap<bool>;

    default_random_engine engine;
    uniform_int_distribution<int> uniform_dist;

    struct Context {
    public:
        G g;
        vector<Node> nodes; // Indexed by file id - 1.
        Cut reference_cut;
        NodeMapi original_ids;
        size_t num_vertices;
        vector<Matchingp> matchings;
        vector<Cutp> cuts;

        explicit Context() : original_ids(g) {
            if (READ_GRAPH_FROM_FILE) {
                parse_chaco_format(IN_GRAPH_FILE, g, nodes, original_ids);

                if (COMPARE_PARTITION) {
                    read_partition_file(PARTITION_FILE, nodes, reference_cut);
                }
            } else {
                cout << "Generating graph with " << N_NODES << " nodes." << endl;
                generate_large_graph(g, nodes);
            }

            num_vertices = countNodes(g);
            assert(num_vertices % 2 == 0);
            assert(connected(g));
        }
    };

    struct MatchingContext {
        G &g;
        Node s;
        Node t;
        EdgeMapi capacity;
        CutMap cut_map;
        const size_t num_vertices;
        Snapshot snap; //RAII

        explicit MatchingContext(Context &c)
                : g(c.g),
                  capacity(g),
                  cut_map(g),
                  snap(g),
                  num_vertices(c.num_vertices)
        { }

        ~MatchingContext() {
            snap.restore();
        }

        bool touches_source_or_sink(Edge &e) {
            return this->g.u(e) == s
                   || this->g.v(e) == s
                   || this->g.u(e) == t
                   || this->g.v(e) == t;
        }

        void copy_cut(Cutp &cut) {
            cut.reset(new Cut);
            for (NodeIt n(this->g); n != INVALID; ++n) {
                if (n == s || n == t) continue;
                if (cut_map[n]) cut->insert(n);
            }
        }

        void reset_cut_map() {
            for (NodeIt n(this->g); n != INVALID; ++n) {
                cut_map[n] = false;
            }
        }
    };

    CutMatching() : uniform_dist(0, 1) {};

    static void read_partition_file(const string &filename, const vector<Node> &nodes, Cut &partition) {
        ifstream file;
        file.open(filename);
        if (!file) {
            cerr << "Unable to read file " << filename << endl;
            exit(1);
        }
        bool b;
        size_t i = 0;
        while (file >> b) {
            if (b) partition.insert(nodes[i]);
            ++i;
        }
        cout << "Reference patition size: " << partition.size() << endl;
    }

    // Soooooo, we want to develop the partition comparison stuff.

    // Actually, cut player gets H
// Actually Actually, sure it gets H but it just needs the matchings...
    template<typename M>
    Bisectionp cut_player(const G &g, const vector<unique_ptr<M>> &matchings) {
        cout << "Running Cut player" << endl;
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
        auto b = Bisectionp(new Bisection(all_nodes.begin(), all_nodes.end()));
        if (PRINT_NODES) { print_cut(*b); }
        return b;
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

    void decompose_paths_fast(const MatchingContext &mg, const unique_ptr<FlowAlgo> &f, Paths &out_paths) {
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

    size_t bin_search_flows(MatchingContext &mg, unique_ptr<FlowAlgo> &p, Cutp &out_cut) const {
        // TODO Output cut
        cout << "Running binary search on flows" << endl;
        auto start = high_resolution_clock::now();

        size_t cap = 1;
        for (; cap < mg.num_vertices; cap *= 2) {
            for (EdgeIt e(mg.g); e != INVALID; ++e) {
                if (mg.touches_source_or_sink(e)) continue;
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
                // Already an expander I guess?
                if (cap == 1) {
                    // TODO code duplication
                    mg.reset_cut_map();
                    p->minCutMap(mg.cut_map);
                }
                break;
            }

            // So it will always have the mincutmap of "before"
            // recomputed too many times of course but whatever
            mg.reset_cut_map();
            p->minCutMap(mg.cut_map);
        }

        // Not we copy out the cut
        mg.copy_cut(out_cut);

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Flow search took (microsec) " << duration.count() << endl;

        return cap;
    }

    void decompose_paths(const MatchingContext &mg, const unique_ptr<FlowAlgo> &p, vector<array<Node, 2>> &paths) {
        cout << "Decomposing paths." << endl;
        auto start = high_resolution_clock::now();
        decompose_paths_fast(mg, p, paths);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Path decomposition took (microsec) " << duration.count() << endl;
    }

// returns capacity that was required
// Maybe: make the binsearch an actual binsearch
    size_t matching_player(Context &c, const set<Node> &bisection, ListEdgeSet<G> &m_out, Cutp &cut) {
        MatchingContext mg(c);
        make_sink_source(mg, bisection);

        unique_ptr<FlowAlgo> p;
        size_t cap_needed = bin_search_flows(mg, p, cut);

        vector<array<Node, 2>> paths;
        decompose_paths(mg, p, paths);

        for (auto &path : paths) {
            m_out.addEdge(path[0], path.back());
        }
        // Now how do we extract the cut?
        // In this version, in one run of the matching the cut is strictly decided. We just need
        // to decide which one of them.
        // Only when we change to edge will the cut need to be explicitly extracted.
        // Rn the important thing is to save cuts between rounds so I can choose the best.

        return cap_needed;
    }

    void make_sink_source(MatchingContext &mg, const set<Node> &cut) const {
        G &g = mg.g;
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

    static void generate_large_graph(G &g, vector<Node>& nodes) {
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

    // Reads the file filename,
    // creates that graph in graph g which is assumed to be empty
    // In the process fills nodes with each node created at the index of (its id in the file minus one)
    // And sets each node's original_ids id to be (its id in the file minus one).
    // Of course original_ids must be initialized onto the graph g already earlier.
    static void parse_chaco_format(const string &filename, ListGraph &g, vector<Node> &nodes, NodeMapi &original_ids) {
        assert(nodes.empty());
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

        for (size_t i = 0; i < n_verts; i++) {
            Node n = g.addNode();
            nodes.push_back(n);
            original_ids[n] = i;
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
            Node n = g.addNode();
            g.addEdge(nodes[0], n);
            nodes.push_back(n);
        }
    }

    size_t one_round(Context &c) {
        Bisectionp bisection = cut_player(c.g, c.matchings);

        Matchingp matchingp(new Matching(c.g));

        cout << "Running Matching player" << endl;
        Cutp cut;
        size_t cap = matching_player(c, *bisection, *matchingp, cut);
        if (PRINT_NODES) { print_matching(matchingp); }

        c.matchings.push_back(move(matchingp));
        c.cuts.push_back(move(cut));
        return cap;

        // We want to implement that it parses partitions
        // That has nothing to do with the rounds lol
    }

    void print_matching(const Matchingp &m) {
        cout << "Matching player gave the following matching: " << endl;
        for (Matching::EdgeIt e(*m); e != INVALID; ++e) {
            cout << "(" << m->id(m->u(e)) << ", " << m->id(m->v(e)) << "), ";
        }
        cout << endl;
    }

    void print_cut(const Bisection &out_cut) const {
        cout << "Cut player gave the following cut: " << endl;
        for (Node n : out_cut) {
            cout << G::id(n) << ", ";
        }
        cout << endl;
    }

    bool is_crossing(const G &g, const Bisection &c, const Edge &e) {
        bool u_in = c.count(g.u(e));
        bool v_in = c.count(g.v(e));
        return u_in != v_in;
    }

    void print_end_round(int i) const {
        cout << "======================" << endl;
        cout << "== End round " << i << endl;
        cout << "======================" << endl;
    }

    void print_cut_sparsity(const Context &c, const Cut &cut) {
        double crossing_edges = 0;
        for (EdgeIt e(c.g); e != INVALID; ++e) {
            if (is_crossing(c.g, cut, e)) crossing_edges += 1;
        }
        assert(cut.size() <= c.num_vertices);
        cout << "Edge crossings (E) : " << crossing_edges << endl;
        size_t other_size = c.num_vertices - cut.size();
        double min_side = min(cut.size(), other_size);
        double max_side = max(cut.size(), other_size);
        double diff = max_side - min_side;
        double factor = diff/c.num_vertices;
        cout << "cut size: (" << min_side << " | " << max_side << ")" << endl
        << "diff: " << diff << " (" << factor << " of total n vertices)" << endl;
        cout << "Min side: " << min_side << endl;
        double expansion_maybe = crossing_edges / min_side;

        cout << "E/min(|S|, |comp(S)|) = " << expansion_maybe << endl;
    }

    size_t run_rounds(Context& c) {
        size_t best_cap = 0;
        size_t best_cap_index = 999999;
        for (int i = 0; i < N_ROUNDS; i++) {
            size_t cap = one_round(c);
            print_end_round(i);

            if (cap > best_cap) {
                best_cap = cap;
                best_cap_index = i;
            }
        }

        return best_cap_index;
    }

    void write_cut(const Context& c, const Cut& cut) {
        ofstream file;
        file.open(OUTPUT_FILE);
        if(!file) {
            cout << "Cannot open file " << OUTPUT_FILE << endl;
            return;
        }

        cout << "Writing partition with "
        << c.nodes.size()
        << " nodes to file "
        << OUTPUT_FILE
        << endl;
        for(const auto& n : c.nodes) {
            file << (cut.count(n) ? "1" : "0") << "\n";
        }
        file.close();
    }

    void run() {
        Context c;
        if(N_ROUNDS >= 1) {
            auto best_round = run_rounds(c);
            cout << "The cut with highest capacity required was found on round" << best_round << endl;
            cout << "Best cut sparsity: " << endl;
            auto& best_cut = *c.cuts[best_round];
            print_cut_sparsity(c, best_cut);
            if(OUTPUT_CUT) { write_cut(c, best_cut); }
        }

        if (COMPARE_PARTITION) { // Output reference cut
            cout << endl
                 << "The given partition achieved the following:"
                 << endl;
            print_cut_sparsity(c, c.reference_cut);
        }
    }
};

cxxopts::Options create_options() {
    cxxopts::Options options("Janiuk graph partition",
                             "Individual project implementation of thatchapon's paper to find graph partitions. Currently only cut-matching game.");
    options.add_options()
            ("f,file", "File to read graph from", cxxopts::value<std::string>())
            ("n,nodes", "Number of nodes in graph to generate. Should be even. Ignored if -f is set.",
             cxxopts::value<long>()->default_value("100"))
            ("r,rounds", "Number of rounds to run cut-matching game", cxxopts::value<long>()->default_value("5"))
            ("d,paths", "Whether to print paths")
            ("v,verbose", "Whether to print nodes and cuts (does not include paths)")
            ("s,seed", "Use a seed for RNG (optionally set seed manually)",
             cxxopts::value<int>()->implicit_value("1337"))
            ("o,output", "Output computed cut into file", cxxopts::value<std::string>())
            ("p,partition", "Partition file to compare with", cxxopts::value<std::string>());
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
    if(result.count("output")) {
        OUTPUT_CUT = true;
        cout << "Got flag for output: " << result["output"].as<string>() << endl;
        OUTPUT_FILE = result["output"].as<string>();
    }
    if (result.count("partition")) {
        COMPARE_PARTITION = true;
        PARTITION_FILE = result["partition"].as<string>();
    }
}

// TODO Selecting best cut not only hightest cap
int main(int argc, char **argv) {
    CutMatching<ListGraph> cm;
    parse_options(argc, argv, cm);
    cm.run();
    return 0;
}

#pragma clang diagnostic pop