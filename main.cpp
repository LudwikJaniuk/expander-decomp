// Authored by Ludvig Janiuk 2019 as part of individual project at KTH.

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-msc32-c"
#pragma ide diagnostic ignored "cppcoreguidelines-slicing"

#include <iostream>
#include <ostream>
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

#define MULTIGRAPH 0
#define REAL_MULTI 1

// To write this with support for multi, need to:
// 1. Parse multi (perhaps map with multiplicities) (map multiplicities and more-volumes of nodes)
//      seems eash.
// 2. Make sure subdiv happens well with multi
//      Need to copy over the map and set vlaues for coped edges. And copied verts. Selfloops.
// DONE
// 3. IS cut_player random walk thing affected? I think not...
// Done, didsnt require anything
// 4. capacities must be set correct with flow
//      Yes indeed, scaled
//      But that means path decomp should be unaffected
// 5. computations of conductance and volume must still be correct
//      Lucky we have cutstats.
// Should be done...
// 6. does multi affect matchings? We don

using namespace lemon;
using namespace std;
using namespace std::chrono;

using G = ListGraph;
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
template<class T> using EdgeMap = typename G::template EdgeMap<T>;
using EdgeMapi = EdgeMap<int>; // LEMON uses ints internally. We might want to look into this
using EdgeMapb = EdgeMap<bool>; // LEMON uses ints internally. We might want to look into this
template<class T> using NodeMap = typename G::template NodeMap<T>;
using NodeMapi = NodeMap<int>;
using NodeMapb= NodeMap<bool>;
using NodeNeighborMap = NodeMap<vector<tuple<Node, int>>>;
using FlowAlgo = Preflow<G, EdgeMapi>;
using Matching = vector<array<Node, 2>>;
using Matchingp = unique_ptr<Matching>;
using Bisection = set<Node>;
using Bisectionp = unique_ptr<Bisection>;
using Cut = set<Node>;
using Cutp = unique_ptr<Cut>;
using CutMap = NodeMap<bool>;

const double MICROSECS = 1000000.0;
const auto& now = high_resolution_clock::now;
double duration_sec(const high_resolution_clock::time_point& start, high_resolution_clock::time_point& stop) {
    return duration_cast<microseconds>(stop - start).count() / MICROSECS;
}

struct InputConfiguration {
    bool load_from_file = false;
    string file_name = "";
    bool ignore_multi = false;
    size_t n_nodes_to_generate;
};

struct Configuration {
    InputConfiguration input;
    bool compare_partition = false;
    string partition_file = "";
    bool seed_randomness = false;
    int seed;
    int max_rounds;
    bool output_cut;
    string output_file;
    bool show_help_and_exit = false;
    bool use_H_phi_target = false;
    double H_phi_target = 0;
    bool use_G_phi_target = false;
    double G_phi_target = 0;
    // we only break if we find a good enough cut that is also this balanced (has this minside volume)
    bool use_volume_treshold = false;
    double volume_treshold_factor = 0;
};

struct Logger {
    bool silent = false;
    bool verbose = false;
    ofstream nul; // UNopened file stream, will act like /dev/null
    Logger() : nul() { };
    decltype(cout)& progress() {
        return silent ? nul : cout ;
    };
    decltype(cout)& debug() {
        return verbose ? cout : nul;
    };
} l;

struct GraphContext {
    G g;
    vector<Node> nodes;
    long num_edges;

#if MULTIGRAPH
    // Used to implement multigraph.
    EdgeMap<size_t> multiplicity;
    // Saving self loops like this
    NodeMap<size_t> num_self_loops;

    GraphContext() :
        g(),
        multiplicity(g, 1),
        num_self_loops(g, 0)
    {}
#endif
};

// I'd say implementing our own adaptor is more effort, we can just do the snapshot thing
// Actually lets just subdivide manually at the start and we dont even need to restore.
struct SubdividedGraphContext {
    SubdividedGraphContext (GraphContext& gc) :
    origContext(gc),
    nf(sub_g),
    ef(sub_g),
    n_ref(gc.g, INVALID), // to_copied
    n_cross_ref(sub_g, INVALID), // from copued
#if MULTIGRAPH
    e_ref(gc.g, INVALID), // to_copied
    e_cross_ref(sub_g, INVALID), // from copued
#endif
    origs(sub_g, false),
    only_splits(sub_g, nf, ef)
#if MULTIGRAPH
            , multiplicity(sub_g, 1),
            num_self_loops(sub_g, 0)
#endif
    {} ;

    GraphContext& origContext;
    G sub_g;
    NodeMapb nf;
    EdgeMapb ef;
#if MULTIGRAPH
    EdgeMap<Edge> e_cross_ref;
    EdgeMap<Edge> e_ref;
#endif
    NodeMap<Node> n_cross_ref;
    NodeMap<Node> n_ref;
    NodeMap<bool> origs;
    SubGraph<G> only_splits;
    vector<Node> split_vertices;

#if MULTIGRAPH
    // Used to implement multigraph.
    EdgeMap<size_t> multiplicity;
    // Saving self loops like this
    NodeMap<size_t> num_self_loops;
#endif
};

// TODO What chnages will be necessary?
struct RoundReport {
    size_t index;
    size_t capacity_required_for_full_flow;
    double multi_h_conductance;
    double g_conductance;
    long volume;
    bool relatively_balanced;
    Cutp cut;
};

template <class G>
struct CutStats {
    using Node = typename G::Node;
    using Edge = typename G::Edge;
    using Cut = set<Node>;
    using EdgeIt = typename G::EdgeIt;
    using Bisection = set<Node>;
    size_t crossing_edges = 0;

private:
    bool is_min_side;
    size_t min_side = 0;
    size_t cut_volume = 0;
    size_t max_side = 0;
#if !REAL_MULTI
    size_t num_edges = 0;
    long degreesum() { return num_edges*2;}
    long noncut_volume () { return degreesum() - cut_volume;}
#else
    size_t cut_size = 0;
    size_t othersize = 0;
    size_t non_cut_volume = 0;
    long noncut_volume () { return non_cut_volume; }
#endif

public:
    CutStats(const G &g, size_t num_vertices, const Cut &cut) {
        initialize(g, num_vertices, cut);
    }

    void initialize(const G &g, size_t num_vertices, const Cut &cut) {
        for (EdgeIt e(g); e != INVALID; ++e) {
#if !REAL_MULTI
            ++num_edges;
#endif
            if (is_crossing(g, cut, e)) crossing_edges += 1;
            if (cut.count(g.u(e))) cut_volume += 1;
#if REAL_MULTI
            else non_cut_volume += 1;
#endif
            // If it's a self loop, if only contributes once, but if not, it contributes twice
            if (g.u(e) == g.v(e)) continue;
            if (cut.count(g.v(e))) cut_volume += 1;
#if REAL_MULTI
            else non_cut_volume += 1;
#endif
        }

        assert(cut.size() <= num_vertices);
        size_t other_size = num_vertices - cut.size();
        min_side = min(cut.size(), other_size);
        max_side = max(cut.size(), other_size);
        is_min_side = cut.size() == min_side;

#if REAL_MULTI
        cut_size = cut.size();
        othersize = other_size;
#endif
    }

    static bool is_crossing(const G &g, const Bisection &c, const Edge &e) {
        bool u_in = c.count(g.u(e));
        bool v_in = c.count(g.v(e));
        return u_in != v_in;
    }

    static bool any_in_cut(const G &g, const Bisection &c, const Edge &e) {
        bool u_in = c.count(g.u(e));
        bool v_in = c.count(g.v(e));
        return u_in || v_in;
    }

    long minside_volume() {
        return is_min_side ? cut_volume : noncut_volume();
    }

    long maxside_volume() {
        return is_min_side ? noncut_volume() : cut_volume;
    }

    size_t diff() {
        return max_side - min_side;
    }

    size_t num_vertices() {
        return min_side + max_side;
    }

    double imbalance() {
        return diff() * 1. / num_vertices();
    }

    double expansion() {
        return min_side == 0 ? 0 : crossing_edges * 1. / min_side;
    }

#if REAL_MULTI
    long min_volume() {
        return min(cut_volume, non_cut_volume);
    }
#endif

double conductance() {
#if REAL_MULTI
        return min_side == 0 ? 999 : crossing_edges * 1. / min_volume();
#else
        // TODO But this is probably wrong in all cases...
        return min_side == 0 ? 999 : crossing_edges * 1. / minside_volume();
#endif
    }



    void print(string prefix="") {
        l.progress() << prefix << "Edge crossings (E) : " << crossing_edges << endl;
#if REAL_MULTI
        l.progress() << prefix << "cut size: ( " << cut_size << " | " << othersize << " )" << endl
             << "diff: " << diff() << " (factor " << imbalance() << " of total n vertices)" << endl;
        l.progress() << prefix << "cut volumes: ( " << cut_volume << " | " << non_cut_volume << " )" << endl;
#else
        l.progress() << prefix << "cut size: (" << min_side << " | " << max_side << ")" << endl
             << "diff: " << diff() << " (factor " << imbalance() << " of total n vertices)" << endl;
        l.progress() << prefix << "Min side: " << min_side << endl;
#endif
        l.progress() << prefix << "expansion: " << expansion() << endl;
        l.progress() << prefix << "conductance: " << conductance() << endl;
#if !REAL_MULTI
        l.progress() << prefix << prefix << "cut volume: " << cut_volume << endl;
        l.progress() << prefix << "noncut volume: " << noncut_volume() << endl;
#endif
    }
};
// Reads the file filename,
// creates that graph in graph g which is assumed to be empty
// In the process fills nodes with each node created at the index of (its id in the file minus one)
// And sets each node's original_ids id to be (its id in the file minus one).
// Of course original_ids must be initialized onto the graph g already earlier.
static void parse_chaco_format
        ( const string &filename
        , ListGraph &g
        , vector<Node> &nodes
        , bool take_multi_edges
#if MULTIGRAPH
        , EdgeMap<size_t>& multiplicity
        , NodeMap<size_t>& num_self_loops
#endif
) {
    assert(nodes.empty());
    l.progress() << "Reading graph from " << filename << endl;
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
    l.progress() << "Reading a graph with V " << n_verts << "E " << n_edges << endl;
    //g.reserveNode(n_verts);
    //g.reserveEdge(n_edges);

    for (size_t i = 0; i < n_verts; i++) {
        Node n = g.addNode();
        nodes.push_back(n);
#if MULTIGRAPH
        num_self_loops[n] = 0;
#endif
    }

    for (size_t i = 0; i < n_verts; i++) {
        getline(file, line);
        istringstream iss(line);
        vector<string> tokens{istream_iterator<string>{iss},
                              istream_iterator<string>{}};
        Node u = nodes[i];
        for(string& str : tokens) {
            size_t v_name = stoi(str);
            l.debug() << "edge from " << i << " to: " << v_name << "..." ;
            assert(v_name != 0);

            Node v = nodes[v_name - 1];

#if REAL_MULTI

            if (take_multi_edges || findEdge(g, u, v)== INVALID) {
                g.addEdge(u, v);
            }
#else // REAL_MULTI

            // TODO So here we would just remove the thing, and see at the end that the numbers match
            // TODO Actually no. We would maintain a map that counts the multiplicity

#if MULTIGRAPH
            // Self loop case, just count up the self loop map
            if (v == u) {
                num_self_loops[u] = num_self_loops[u]+1;
                continue;
            }
#endif

            Edge found_edge = findEdge(g, u, v);

            if (found_edge == INVALID) {
                l.debug()  << "adding" << endl;
                found_edge = g.addEdge(u, v);
#if MULTIGRAPH
                multiplicity[found_edge] = 0;
#endif
            }

#if MULTIGRAPH
            multiplicity[found_edge] = multiplicity[found_edge]+1;
#endif
#endif // REAL_MULTI

        }

    }
    cout << countEdges(g) << endl;
    cout << countNodes(g) << endl;
    assert(countEdges(g) == n_edges);
    assert(countNodes(g) == n_verts);

#if MULTIGRAPH
    // Count for assert
    int counted_edges = 0;
    int counted_verts = 0;
    for(EdgeIt e(g); e != INVALID; ++e) {
        assert(multiplicity[e] > 0);
        counted_edges += multiplicity[e];
    }
    for(NodeIt n(g); n != INVALID; ++n) {
        counted_verts++;
        counted_edges += num_self_loops[n];
    }
    assert(n_edges == counted_edges);
    assert(n_verts == counted_verts);
#endif
}

void generate_large_graph(G &g, vector<Node> &nodes, size_t n_nodes) {
    assert(n_nodes > 0);
    nodes.reserve(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        nodes.push_back(g.addNode());
    }

    g.addEdge(nodes[0], nodes[1]);
    g.addEdge(nodes[1], nodes[2]);
    g.addEdge(nodes[2], nodes[0]);

    int lim1 = n_nodes / 3;
    int lim2 = 2 * n_nodes / 3;

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
    for (int i = lim2; i < n_nodes; i++) {
        ListGraph::Node u = nodes[i];
        ListGraph::Node v = nodes[2];
        g.addEdge(u, v);
    }
}

void write_cut(const vector<Node> &nodes, const Cut &cut, string file_name) {
    ofstream file;
    file.open(file_name);
    if (!file) {
        l.progress() << "Cannot open file " << file_name << endl;
        return;
    }

    l.debug() << "Writing partition with "
         << nodes.size()
         << " nodes to file "
         << file_name
         << endl;
    for (const auto &n : nodes) {
        file << (cut.count(n) ? "1" : "0") << "\n";
    }
    file.close();
}

void read_partition_file(const string &filename, const vector<Node> &nodes, Cut &partition) {
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
    l.debug() << "Reference patition size: " << partition.size() << endl;
}

void initGraph(GraphContext &gc, InputConfiguration config) {
    if (config.load_from_file) {
#if MULTIGRAPH
        parse_chaco_format(config.file_name, gc.g, gc.nodes, gc.multiplicity, gc.num_self_loops);
#else
        parse_chaco_format(config.file_name, gc.g, gc.nodes, !config.ignore_multi);
#endif

    } else {
        l.debug() << "Generating graph with " << config.n_nodes_to_generate << " nodes." << endl;
        generate_large_graph(gc.g, gc.nodes, config.n_nodes_to_generate);
    }

    gc.num_edges = countEdges(gc.g);
    l.debug() << "gc.num_edges: " << gc.num_edges << endl;
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

void print_end_round_message(int i) {
    l.debug() << "======================" << endl;
    l.progress() << "== End round " << i << " ==" << endl;
    l.debug() << "======================" << endl;
}

void print_cut(const Bisection &out_cut, decltype(cout)& stream) {
    for (Node n : out_cut) {
        stream << G::id(n) << ", ";
    }
    stream << endl;
}

void print_graph(G& g, decltype(cout)& stream) {
    stream << "Printing a graph" << endl;
    stream << "Vertices: " << countNodes(g) << ", Edges: " << countEdges(g) << endl;
    stream << "==" << endl;
    for(NodeIt n(g); n != INVALID; ++n) {
        stream << g.id(n) << ", ";
    }
    stream << "\n==" << endl;
    for(EdgeIt e(g); e != INVALID; ++e) {
        stream << g.id(e) << ": " << g.id(g.u(e)) << " - " << g.id(g.v(e)) << "\n";
    }
    stream << endl;
}


// Actually copies the graph.
void createSubdividedGraph(SubdividedGraphContext& sgc) {
    graphCopy(sgc.origContext.g, sgc.sub_g)
      .nodeRef(sgc.n_ref)
      .nodeCrossRef(sgc.n_cross_ref)
#if MULTIGRAPH
      .edgeRef(sgc.e_ref)
      .edgeCrossRef(sgc.e_cross_ref)
#endif
      .run();
    G& g = sgc.sub_g;
    for (NodeIt n(g); n != INVALID; ++n) {
        sgc.origs[n] = true;
#if MULTIGRAPH
        sgc.num_self_loops[n] = sgc.origContext.num_self_loops[sgc.n_cross_ref[n]];
#endif
    }

    // Too bad we do a copy cuz then we kinda need to copy the map too?
    // TODO either copy the edge map, or need to add an edge crossref. Does that exist?
    // Yes it does! http://lemon.cs.elte.hu/pub/doc/latest/a00177.html#a49664d231463c080f54ebd3299cd1121
    // Except we're deleting the edges. So maybe lets copy the map instead, cleaner code and faster.
    // THe right engineering decision.

    vector<Edge> edges;
    for (EdgeIt e(g); e != INVALID; ++e) {
        edges.push_back(e);
    }

    for (NodeIt n(g); n != INVALID; ++n) {
        sgc.only_splits.disable(n);
    }

    for(auto& e : edges) {
        Node u = g.u(e);
        Node v = g.v(e);
#if MULTIGRAPH
        size_t edge_multiplicity = sgc.origContext.multiplicity[sgc.e_cross_ref[e]];
#endif
        g.erase(e);

        Node s = g.addNode();
#if MULTIGRAPH
        sgc.num_self_loops[s] = 0;
#endif
        sgc.origs[s] = false;
        sgc.only_splits.enable(s);

#if MULTIGRAPH
        assert(u != v);
        Edge e1 = g.addEdge(u, s);
        sgc.multiplicity[e1] = edge_multiplicity;
        Edge e2 = g.addEdge(s, v);
        sgc.multiplicity[e2] = edge_multiplicity;
        // Trying not to have this become a multigraph
#else
        // This behavior is correct with real_multi assuming (1)
        g.addEdge(u, s);
        // Trying not to have this become a multigraph
        if(u != v) g.addEdge(s, v);
#endif
        sgc.split_vertices.push_back(s);
    }
}

struct CutMatching {
    const Configuration &config;
    GraphContext &gc;
    SubdividedGraphContext sgc;
    default_random_engine &random_engine;
    //vector<unique_ptr<RoundReport>> past_rounds;
    vector<unique_ptr<RoundReport>> sub_past_rounds;
    vector<Matchingp> matchings;
    vector<Matchingp> sub_matchings;
    bool reached_H_target = false;
    // Input graph
    CutMatching(GraphContext &gc, const Configuration &config_, default_random_engine &random_engine_)
    :
    config(config_),
    gc(gc),
    sgc{gc},
    random_engine(random_engine_)
    {
        //assert(gc.nodes.size() % 2 == 0);
        assert(gc.nodes.size() > 0);
        assert(connected(gc.g));

        l.debug() << "got graph: " << endl;
        print_graph(gc.g, l.debug());
        createSubdividedGraph(sgc);
        l.debug() << "subdivided into: " << endl;
        print_graph(sgc.sub_g, l.debug());
    };

    // During the matching step a lot of local setup is actually made, so it makes sense to group it
    // inside a "matching context" that exists for the duration of the mathing step
    struct MatchingContext {
        // This NEEDS to be the whole graph
        G& g;
        Node s;
        Node t;
        EdgeMapi capacity;
        CutMap cut_map;
        Snapshot snap; //RAII

        explicit MatchingContext(G& g_)
        :
        g(g_),
        capacity(g_),
        cut_map(g_),
        snap(g_)
        {}

        ~MatchingContext() {
            snap.restore();
        }

        bool touches_source_or_sink(Edge &e) {
            return g.u(e) == s
                   || g.v(e) == s
                   || g.u(e) == t
                   || g.v(e) == t;
        }

        // Fills given cut pointer with a copy of the cut map
        Cutp extract_cut() {
            Cutp cut(new Cut);
            for (NodeIt n(g); n != INVALID; ++n) {
                if (n == s || n == t) continue;
                if (cut_map[n]) cut->insert(n);
            }
            return move(cut);
        }

        void reset_cut_map() {
            for (NodeIt n(g); n != INVALID; ++n) {
                cut_map[n] = false;
            }
        }
    };

    struct MatchResult {
        Cutp cut_from_flow;
        size_t capacity; // First capacity (minumum) that worked to get full flow thru
    };

    inline void extract_path_fast(
            const G &g,
            const unique_ptr<Preflow<G, EdgeMapi>> &f,
            NodeNeighborMap &flow_children,
            Node u_orig,
            Node t, // For assertsions
            array<Node, 2> &out_path
    ) {
        out_path[0] = u_orig;
        Node u = u_orig;
        while (true) {

            auto& vv = flow_children[u];
            assert(vv.size() > 0);
            auto &tup = vv.back();
            Node v = get<0>(tup);
            --get<1>(tup);

            if (get<1>(tup) == 0) flow_children[u].pop_back();

            if (flow_children[v].empty()) {
                assert(v == t);
                assert(u != u_orig);

                out_path[1] = u;
                break;
            }

            u = v;
        }
    }

    // TODO This will be different with multi
    // OTOH if we're going maps, then it won't.
    void decompose_paths_fast(const MatchingContext &mg, const unique_ptr<FlowAlgo> &f, Paths &out_paths) {
        f->startSecondPhase();
        EdgeMapi subtr(mg.g, 0);
        NodeNeighborMap flow_children(mg.g, vector<tuple<Node, int>>());
        out_paths.reserve(countNodes(mg.g) / 2);

        // Calc flow children (one pass)
        ArcLookup alp(mg.g);
        for (EdgeIt e(mg.g); e != INVALID; ++e) {
            Node u = mg.g.u(e);
            Node v = mg.g.v(e);
            long e_flow = flow(alp, f, u, v);

            l.debug() << "FLOW " << mg.g.id(u) << " -> " << mg.g.id(v) << " : " << e_flow << endl;
            if (e_flow > 0) {
                flow_children[u].push_back(tuple(v, e_flow));
            } else if (e_flow < 0) {
                flow_children[v].push_back(tuple(u, -e_flow));
            }
        }

        for (IncEdgeIt e(mg.g, mg.s); e != INVALID; ++e) {
            assert(mg.g.u(e) == mg.s || mg.g.v(e) == mg.s);
            Node u = mg.g.u(e) == mg.s ? mg.g.v(e) : mg.g.u(e);

            // This approach would not produce a matching in the end...
            //if(e_flow == 0) {
            //    cout << "SKIPPING" << endl;
            //    continue;
           // }

            out_paths.push_back(array<Node, 2>());
            extract_path_fast(mg.g, f, flow_children, u, mg.t, out_paths[out_paths.size() - 1]);
        }
    }

    // Works for sub too, with the assumption that mg.g realy is the whole graph
    void run_min_cut(const MatchingContext &mg, unique_ptr<FlowAlgo> &p) const {
        p.reset(new Preflow<G, EdgeMapi>(mg.g, mg.capacity, mg.s, mg.t));
        auto start2 = now();
        p->runMinCut(); // Note that "startSecondPhase" must be run to get flows for individual verts
        auto stop2 = now();
        l.progress() << "flow: " << p->flowValue() << " (" << duration_sec(start2, stop2) << " s)" << endl;
    }

    // This should work well for sub too
    void set_matching_capacities(MatchingContext &mg, size_t cap) const {
        for (EdgeIt e(mg.g); e != INVALID; ++e) {
            if (mg.touches_source_or_sink(e)) continue;
            mg.capacity[e] = cap;
        }
    }

    MatchResult bin_search_flows(MatchingContext &mg, unique_ptr<FlowAlgo> &p, size_t flow_target) const {
        auto start = now();
        size_t cap = 1;
        // TODO will need to set capacities scaled by multiplicity
        //for (; cap < mg.gc.nodes.size(); cap *= 2) {
        for (; cap < flow_target*2; cap *= 2) {
            l.progress() << "Cap " << cap << " ... " << flush;
            set_matching_capacities(mg, cap);
            run_min_cut(mg, p);

            //bool reachedFullFlow = p->flowValue() == mg.gc.nodes.size() / 2;
            bool reachedFullFlow = p->flowValue() >= flow_target;
            if (reachedFullFlow) l.debug() << "We have achieved full flow, but half this capacity didn't manage that!" << endl;

            // So it will always have the mincutmap of "before"
            // mincuptmap is recomputed too many times of course but whatever
            // If we reached it with cap 1, already an expander I guess?
            // In this case this was never done even once, so we have to do it before breaking
            if (!reachedFullFlow || cap == 1) {
                mg.reset_cut_map();
                p->minCutMap(mg.cut_map);
            }

            if (reachedFullFlow) break;
        }

        // Not we copy out the cut
        MatchResult result{mg.extract_cut(), cap};

        auto stop = now();
        l.progress() << "Flow search took (seconds) " << duration_sec(start, stop) << endl;

        return result;
    }

    void decompose_paths(const MatchingContext &mg, const unique_ptr<FlowAlgo> &p, vector<array<Node, 2>> &paths) {
        decompose_paths_fast(mg, p, paths);
    }

    // TODO this is totally wrong for edge version
    template <typename GG>
    void make_sink_source(GG& g, MatchingContext &mg, const set<Node> &cut) const {
        mg.s = g.addNode();
        mg.t = g.addNode();
        int s_added = 0;
        int t_added = 0;
        for (typename GG::NodeIt n(g); n != INVALID; ++n) {
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
        int diff = s_added - t_added;
        assert(-1 <= diff && diff <= 1);
    }



    // Actually, cut player gets H
// Actually Actually, sure it gets H but it just needs the matchings...
// TODO Ok so can we just call this with split_only and matchings of those?
    template<typename GG, typename M>
    Bisectionp cut_player(const GG &g, const vector<unique_ptr<M>> &given_matchings, double &h_multi_cond_out) {
        l.debug() << "Running Cut player" << endl;
        typename GG::template NodeMap<double> probs(g);
        vector<Node> all_nodes;

        // Previous bug: would assign 1/"running total". But should just be 1/num verts.
        for (typename GG::NodeIt n(g); n != INVALID; ++n) {
            all_nodes.push_back(n);
        }
        size_t deb_n_nodes = countNodes(g);
        size_t deb_assigned = 0;
        uniform_int_distribution<int> uniform_dist(0, 1);
        for (typename GG::NodeIt n(g); n != INVALID; ++n) {
            probs[n] = uniform_dist(random_engine)
                       ? 1.0 / all_nodes.size()
                       : -1.0 / all_nodes.size();
            deb_assigned++;
        }

        //cout << "DEB: " << deb_n_nodes << " but assigned " << deb_assigned << endl;

        size_t num_vertices = all_nodes.size();

        // Unclear whether H conductance should be computed with its multigraphness in mind, so we're creating both.
        ListEdgeSet H(g);
        ListEdgeSet H_single(g);
        for (const unique_ptr<M> &m : given_matchings) {
            for (auto& e : *m) {
                Node u = e[0];
                Node v = e[1];
                double avg = probs[u] / 2 + probs[v] / 2;
                probs[u] = avg;
                probs[v] = avg;

                H.addEdge(u, v);
                // Updating H_single
                if(findEdge(H_single, u, v) == INVALID) {
                    assert(findEdge(H_single, v, u) == INVALID);
                    H_single.addEdge(u, v);
                }
            }
        }

        // I need to shuffle before sorting to avoid dependency on original order
        // Sort is probably stable and we have many equalis

        shuffle(all_nodes.begin(), all_nodes.end(), random_engine);

        sort(all_nodes.begin(), all_nodes.end(), [&](Node a, Node b) {
            return probs[a] < probs[b];
        });

        size_t size = all_nodes.size();
        // With subdivisions, won't be this way longer
        //assert(size % 2 == 0);
        all_nodes.resize(size / 2);
        auto b = Bisectionp(new Bisection(all_nodes.begin(), all_nodes.end()));
        l.debug() << "Cut player gave the following cut: " << endl;
        print_cut(*b, l.debug());

        // So how does it give output?
        // Ok it assigns h_outs, but actually also returns Bisectionp
        auto hcs = CutStats<decltype(H)>(H, num_vertices, *b);
        l.progress() << "H expansion: " << hcs.expansion() << ", num cross: " << hcs.crossing_edges << endl;
        l.progress() << "H conductance: " << hcs.conductance() << ", num cross: " << hcs.crossing_edges << endl;
        h_multi_cond_out = hcs.conductance();
        auto hscs = CutStats<decltype(H_single)>(H_single, num_vertices, *b);
        l.progress() << "H_single expansion: " << hscs.expansion() << ", num cross: " << hscs.crossing_edges << endl;
        l.progress() << "H_single conductance: " << hscs.conductance() << ", num cross: " << hscs.crossing_edges << endl;

        return b;
    }

    // returns capacity that was required
// Maybe: make the binsearch an actual binsearch
// TODO Let listedgeset just be 2-arrays of nodes. Lemon is getting in the way too much.
// But also what is assigned in MtchResult?
    MatchResult sub_matching_player(const set<Node> &bisection, vector<array<Node, 2>>& m_out) {
        MatchingContext mg(sgc.sub_g);
        make_sink_source(sgc.only_splits, mg, bisection);
        // TODO so have s, t been created on the big greaph?
        Node s = mg.s;
        int id = sgc.sub_g.id(s);
        //cout << id << endl;
        //cout << countNodes(sgc.sub_g) << endl;

        unique_ptr<FlowAlgo> p;
        MatchResult mr = bin_search_flows(mg, p, sgc.split_vertices.size()/2);

        decompose_paths(mg, p, m_out);

        // Now how do we extract the cut?
        // In this version, in one run of the matching the cut is strictly decided. We just need
        // to decide which one of them.
        // Only when we change to edge will the cut need to be explicitly extracted.
        // Rn the important thing is to save cuts between rounds so I can choose the best.

        return mr;
    }

    long sub_volume_treshold() {
        return config.volume_treshold_factor * sgc.origContext.nodes.size();
    }

    // Ok lets attack from here
    // Theres a lot of risk for problems with "is this a cut in the orig graph or in the splits?
    unique_ptr<RoundReport> sub_one_round() {
        unique_ptr<RoundReport> report = make_unique<RoundReport>();

        Bisectionp sub_bisection = cut_player(sgc.only_splits, sub_matchings, report->multi_h_conductance);

        Matchingp smatchingp(new Matching());
        MatchResult smr = sub_matching_player(*sub_bisection, *smatchingp);
        sub_matchings.push_back(move(smatchingp));

        //c.cuts.push_back(move(mr.cut_from_flow));
        report->index = sub_past_rounds.size();
        report->capacity_required_for_full_flow = smr.capacity;
        report->cut = make_unique<Cut>();


        /*
        cout << "cff size " << smr.cut_from_flow->size() << endl;
        cout << "ff Has 0? " << smr.cut_from_flow->count(sgc.n_ref[gc.nodes[0]]) << endl;
        cout << "it is " << sgc.sub_g.id(sgc.n_ref[gc.nodes[0]]) << endl;
         */

        for(auto& n : *(smr.cut_from_flow)) {
           /*
            if(sgc.sub_g.id(n) == sgc.sub_g.id(sgc.n_ref[gc.nodes[0]])) {
                cout << "exists" << endl;
            }
            if(sgc.n_cross_ref[n] == gc.nodes[0]) {
                cout << "imporster exists" << endl;
                cout << sgc.sub_g.id(n) << endl;
                cout << "mapepd: " << sgc.origContext.g.id(sgc.n_cross_ref[n]) << endl;
                cout << "mapepd: " << sgc.origContext.g.id(gc.nodes[0]) << endl;
                cout << "orig: " << sgc.origs[n] << endl;
            }
            */
            //if(sgc.n_cross_ref[n] != INVALID) {
            if(sgc.origs[n]) {
                /*
                if(sgc.sub_g.id(n) == sgc.sub_g.id(sgc.n_ref[gc.nodes[0]])) {
                    cout << "added" << endl;
                }
                if(sgc.n_cross_ref[n] == gc.nodes[0]) {
                    cout << "imporster added" << endl;
                    cout << sgc.sub_g.id(n) << endl;
                }
                 */
                report->cut->insert(sgc.n_cross_ref[n]);
            }
        }

        auto cs = CutStats<G>(sgc.origContext.g, sgc.origContext.nodes.size(), *report->cut);
        report->g_conductance = cs.conductance();
        if(report->g_conductance == 1) {
            cout << "LYING" << endl;
        }
        l.progress() << "SUBG cut conductance: " << report->g_conductance << endl;
        report->volume = cs.minside_volume();
        l.progress() << "SUBG cut minside volume " << cs.minside_volume() << endl;
        l.progress() << "SUBG cut maxside volume " << cs.maxside_volume() << endl;
        report->relatively_balanced = report->volume > sub_volume_treshold();
        return move(report);
    }

    // Stopping condition
    bool sub_should_stop() {
        int i = sub_past_rounds.size();
        if(i == 0) return false;
        if(i >= config.max_rounds && config.max_rounds != 0) return true;

        const auto& last_round = sub_past_rounds[sub_past_rounds.size() - 1];
        if(config.use_H_phi_target && last_round->multi_h_conductance >= config.H_phi_target) {
            l.progress() << "H Conductance target reached, this will be case 1 or 3. According to theory, this means we probably won't find a better cut. That is, assuming you set H_phi right. "
                    "If was used together with G_phi target, this also certifies the input graph is a G_phi expander unless there was a very unbaanced cut somewhere, which we will proceed to look for." << endl;
            reached_H_target = true;
            return true;
        }

        if(config.use_G_phi_target)
            if(last_round->g_conductance <= config.G_phi_target) {
                if(config.use_volume_treshold && last_round->relatively_balanced) {
                    l.progress() << "CASE2 G Expansion target reached with a cut that is relatively balanced. Cut-matching game has found a balanced cut as good as you wanted it."
                         << endl;
                    l.progress() << "Claimed g conductance: " << last_round->g_conductance << endl;
                    return true;
                }

                if(!config.use_volume_treshold) {
                    l.progress() << "G Expansion target reached. Cut-matching game has found a cut as good as you wanted it. Whether it is balanced or not is up to you."
                         << endl;
                    return true;
                }
            }
    }

    void run() {
        while (!sub_should_stop()) {
            //past_rounds.push_back(one_round());
            sub_past_rounds.push_back(sub_one_round());
            print_end_round_message(sub_past_rounds.size()-1);
        }
    }
};

// TODO Make cut always the smallest (maybe)
// TODO (In edge version) Implement breaking-logik for unbalance
// om vi hittar phi-cut med volym obanför treshold
// Om vi hittar phi-cut med volum under treshold, så ingorerar vi det och kör p
// och sen om vi når H, då definieras det bästa som phi-cuttet med högsta volym
cxxopts::Options create_options() {
    cxxopts::Options options("executable_name",
                             "Individual project implementation of thatchapon's paper to find graph partitions. Currently only cut-matching game. \
                             \nRecommended usage: \n\texecutable_name -s -f ./path/to/graph -o output.txt\
                             \nCurrently only running a fixed amount of rounds is supported, but the more correct \
                             \nversion of running until H becomes an expander is coming soon.\
                             ");
    options.add_options()
            ("h,help", "Show help")
            ("H_phi", "Phi expansion treshold for the H graph. Recommend to also set -r=0. ",
             cxxopts::value<double>()->implicit_value("10.0"))
            ("G_phi", "Phi expansion target for the G graph. Means \"what is a good enough cut?\" Recommended with -r=0. This is the PHI from the paper. ",
             cxxopts::value<double>()->implicit_value("0.8"))
            ("vol", "Volume treshold. Only used if G_phi is used. Will be multiplied by number of edges, so to require e.g. minimum 10% volume, write 0.1.",
             cxxopts::value<double>()->implicit_value("0.1"))
            ("f,file", "File to read graph from", cxxopts::value<std::string>())
            ("r,max-rounds", "Number of rounds after which to stop (0 for no limit)", cxxopts::value<long>()->default_value("25"))
            ("s,seed", "Use a seed for RNG (optionally set seed manually)",
             cxxopts::value<int>()->implicit_value("1337"))
            ("p,partition", "Partition file to compare with", cxxopts::value<std::string>())
            ("o,output", "Output computed cut into file. The cut is written as the vertices of one side of the cut.", cxxopts::value<std::string>())
            ("n,nodes", "Number of nodes in graph to generate. Should be even. Ignored if -f is set.",
             cxxopts::value<long>()->default_value("100"))
            ("S,Silent", "Only output one line of summary at the end")
            ("v,verbose", "Debug; Whether to print nodes and cuts Does not include paths. Produces a LOT of output on large graphs.")
            ("ignore-multi", "ignores the same edges repeated when parsing.")
            ("d,paths", "Debug; Whether to print paths")
            ;
    return options;
}

void parse_options(int argc, char **argv, Configuration &config) {
    auto cmd_options = create_options();
    auto result = cmd_options.parse(argc, argv);

    if (result.count("help")) {
        config.show_help_and_exit = true;
        cout << cmd_options.help() << endl;
    }
    if( result.count("H_phi")) {
        config.use_H_phi_target = true;
        config.H_phi_target = result["H_phi"].as<double>();
    }
    if( result.count("G_phi")) {
        config.use_G_phi_target = true;
        config.G_phi_target = result["G_phi"].as<double>();
    }
    if( result.count("vol")) {
        config.use_volume_treshold = true;
        config.volume_treshold_factor = result["vol"].as<double>();
    }
    if (result.count("file")) {
        config.input.load_from_file = true;
        config.input.file_name = result["file"].as<string>();
    }
    if (result.count("nodes"))
        assert(!config.input.load_from_file);
        config.input.n_nodes_to_generate = result["nodes"].as<long>();
    if (result.count("max-rounds"))
        config.max_rounds = result["max-rounds"].as<long>();
    if (result.count("verbose"))
        l.verbose = result["verbose"].as<bool>();
    if (result.count("Silent"))
        l.silent = result["Silent"].as<bool>();

    if (result.count("seed")) {
        config.seed_randomness = true;
        config.seed = result["seed"].as<int>();
    }

    if (result.count("output")) {
        config.output_cut = true;
        config.output_file = result["output"].as<string>();
    }
    if (result.count("ignore-multi")) {
        config.input.ignore_multi = true;
    }

    if (result.count("partition")) {
        config.compare_partition = true;
        config.partition_file = result["partition"].as<string>();
    }
}

int main(int argc, char **argv) {
    Configuration config;
    parse_options(argc, argv, config);

    if(config.show_help_and_exit) return 0;

    GraphContext gc;
    // So right off the bat here, do we want to create the subdivision?
    // No. Let CutMatching do it internally.
    initGraph(gc, config.input);

    default_random_engine random_engine = config.seed_randomness
                    ? default_random_engine(config.seed)
                    : default_random_engine(random_device()());

    // MULTI we are here..
    CutMatching cm(gc, config, random_engine);
    cm.run();

    assert(!cm.sub_past_rounds.empty());
    // Best by conductance
    auto& best_round = *min_element(cm.sub_past_rounds.begin(), cm.sub_past_rounds.end(), [](auto &a, auto &b) {
        return a->g_conductance < b->g_conductance;
    });

    for(int i = 0; i < cm.sub_past_rounds.size(); i++) {
        l.progress() << "R" << i << " cond " << cm.sub_past_rounds[i]->g_conductance << endl;
    }

    l.progress() << "The best with best expansion was found on round" << best_round->index << endl;
    auto &best_cut = best_round->cut;
    CutStats<G>(gc.g, gc.nodes.size(), *best_cut).print("final_");

    if(config.use_H_phi_target && config.use_G_phi_target && config.use_volume_treshold) {
        if(cm.reached_H_target) {

            if(best_round->g_conductance > config.G_phi_target) {
                l.progress() << "CASE1 NO Goodenough cut, G certified expander." << endl;
            } else {
                l.progress() << "CASE3 Found goodenough but very unbalanced cut." << endl;
            }
        } else {
            l.progress() << "CASE2 Goodenough balanced cut" << endl;
        }

    }

    if (config.output_cut) { write_cut(gc.nodes, *best_cut, config.output_file); }

    if (config.compare_partition) {
        Cut reference_cut;
        read_partition_file(config.partition_file, gc.nodes, reference_cut);

        l.progress() << endl
             << "The given partition achieved the following:"
             << endl;
        CutStats<G>(gc.g, gc.nodes.size(), reference_cut).print();
    }

    return 0;
}

#pragma clang diagnostic pop
