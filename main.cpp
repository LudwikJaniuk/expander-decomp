// Authored by Ludvig Janiuk 2019/2020 as part of individual project at KTH.

// See algorithm implementation notes by the CutMatching struct

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

using namespace lemon;
using namespace std;
using namespace std::chrono;


// The following are basically some typedefs to make code easier to read.

// ListGraphs used throughout, seem to be the best choice among LEMON, but others might be better who knows
using G = ListGraph;

// Following shold be self-explanatory
using NodeMapd = typename G::template NodeMap<double>;
using Node = typename G::Node;
using NodeIt = typename G::NodeIt;
using Snapshot = typename G::Snapshot;
using Edge = typename G::Edge;
using EdgeIt = typename G::EdgeIt;
using IncEdgeIt = typename G::IncEdgeIt;
using OutArcIt = typename G::OutArcIt;
using ArcLookup = ArcLookUp<G>;

// Took lots of template magic to make this work... not C++' brightest moment
template<class T> using EdgeMap = typename G::template EdgeMap<T>;
template<class T> using NodeMap = typename G::template NodeMap<T>;
using EdgeMapi = EdgeMap<int>;
using EdgeMapb = EdgeMap<bool>;
using NodeMapi = NodeMap<int>;
using NodeMapb = NodeMap<bool>;

// For path decompositions. For each node, a list of (child, flowToChild) tuples.
using NodeNeighborMap = NodeMap<vector<tuple<Node, int>>>;

// Preflow is the fastest, EdmondsKarp is an alternative which provides more control, but we don't need that control
using FlowAlgo = Preflow<G, EdgeMapi>;

// A matching is a pairing-up of nodes.
using Matching = vector<array<Node, 2>>;
// p suffix implies unique pointer. This pattern is used throughout. LEMON graphs explicitly forbid copying, so this is
// often the only way to move them around (with std::move). And it's useful not only for graphs.
using Matchingp = unique_ptr<Matching>;

// A Bisection represents a perfectly balanced cut. The context needs to keep track of "the whole set" of nodes,
// of course.
using Bisection = set<Node>;
using Bisectionp = unique_ptr<Bisection>;

// A cut is a subset of some larger set of nodes. This data structure does not track thar larger set.
using Cut = set<Node>;
using Cutp = unique_ptr<Cut>;

// A different way to represent a cut.
using CutMap = NodeMap<bool>;

// This used to be a vector of vectors but when optimizing the path decomposition, of course we only need the endpoints
// in the end. Yes, it's now equivalent to a matching. Could in principle be removed.
using Paths = vector<array<Node, 2>>;


// Microseconds per second
const double MICROSECS = 1000000.0;

// Just a reference to shorten code
const auto& now = high_resolution_clock::now;


// Used for printing timing information. Duration in seconds.
double duration_sec(const high_resolution_clock::time_point start, high_resolution_clock::time_point stop) {
    return duration_cast<microseconds>(stop - start).count() / MICROSECS;
}

// To use for stopping running when too much time has passed
struct JaniukException : public std::exception
{
  const char * what () const throw ()
  {
    return "C++ Exception";
  }
};

// Configuration and InputConfiguration just contain all the configuration options.
// The option parsing code produces an instance of Configuration, which then impacts everything.
struct InputConfiguration {
    // Whether to load graph from file, in-program generation is also technically supported
    bool load_from_file = false;
    string file_name = "";
    // Whether to silently ignore repeated edges in the input. This is necessary for correctly reading Walshaw's graphs.
    bool ignore_multi = false;
    // For in-program generation, there's some other switches somewhere else as to type of graph to generate but use is
    // generally discouraged.
    size_t n_nodes_to_generate;
};

struct Configuration {
    // No special reason for splitting these up
    InputConfiguration input;

    // Whether to also load the partition file of this graph (as provided by walshaw) and report stats on that later
    bool compare_partition = false;
    string partition_file = "";

    // Thanks to seeding we can have randomness but still reproducible results. For production use (whatever that would
    // be) you would turn it off.
    bool seed_randomness = false;
    int seed;

    // End after this many rounds, set to 0 to never end because of rounds number.
    int max_rounds;

    // Whether to output the found best cut and into which file
    bool output_cut;
    string output_file;

    bool show_help_and_exit = false;

    // Whether to measure conductance on the cuts on the H graph and end when it goes OVER H_phi_target.
    bool use_H_phi_target = false;
    double H_phi_target = 0;

    // Whether to measure conductance of the cuts on the G graph, and end when we find a cut BELOW this target
    bool use_G_phi_target = false;
    double G_phi_target = 0;
    // we only break if we find a good enough cut that is also this balanced (has this minside volume)
    bool use_volume_treshold = false;
    double volume_treshold_factor = 0;

    // If zero, don't time out.
    float timeout_after_minutes = 0;
};

// Some attempt at nice controllable levels of logging. I'm not really that proud of it.
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


// This idea of "contexts" is trying to strike a balance between avoiding globals, passing too many parameters between
// several layers of functions, and only giving parts of the code as much data as they reasonably need to minimize area
// for bugs. Ideally they should be way more polished and thought out.

// In any case, this is the most basic, just wrapping a graph to cache the number of edges and keep the nodes in an
// easily accessable and passable list.
struct GraphContext {
    G g;
    vector<Node> nodes;
    long num_edges; // Yes long is probably overkill.
};


// An integral part of the algorithm is generating a subdivision graph. The subivision graph is implemented as a full
// copy, and crossreference maps are maintained to know which nodes in each copy correspond to which in the other.
// Not the best in terms of memory footprint, could be implemented with some smarter structures instead and probably
// LEMON adapters. But, footprint is not our primary optimisation concern now.
struct SubdividedGraphContext {
    SubdividedGraphContext (GraphContext& gc) :
     origContext(gc),
     nf(sub_g),
     ef(sub_g),
     n_ref(gc.g, INVALID), // to_copied
     n_cross_ref(sub_g, INVALID), // from copied
     origs(sub_g, false),
     only_splits(sub_g, nf, ef)
    {} ;

    GraphContext& origContext; // The original context, and in it, the original graph.
    G sub_g; // The copy that is the subdivision graph
    NodeMapb nf; // Node filter and
    EdgeMapb ef; // Edge filter directly for use in the only_splits subgraph (LEMON requires such structures to say how
                 // the subgraph shall look.
    NodeMap<Node> n_cross_ref; // Map over the split graph to the corr. vertices in original graph, or INVALID for split
                               // vertices.
    NodeMap<Node> n_ref; // Map over the original graph to the corr. vertices in the split graph
    NodeMap<bool> origs; // Whether a node was one of the original ones, or is a split node.
                         // Inverse of nf.
    SubGraph<G> only_splits; // Adapter with only the split nodes. Can be passed to the cut player.
    vector<Node> split_vertices; // List of the split vertices
};

// Generated after every round, so we keep a growing list and then we can iterate over them and compare at the end.
// Mostly, the cuts are important.
struct RoundReport {
    size_t index; // Zero-indexed.

    size_t capacity_required_for_full_flow; // What's the highest capacity the flow search algorithm had to set.
    // Was used in some earlier versions as a stopping condition.

    double multi_h_conductance;
    double g_conductance;
    long volume; // Always volume of the smaller side
    bool relatively_balanced; // You'll have to look at where it's assigned for exact definition. Used as an additional
    // stopping condition to set a minimum required balancedness.
    Cutp cut;
};

// Many measurements on cuts can be derived from others, so the way you go is you crete a "cutStats" on your cut which
// computers several shared values, and can then be queried for e.g. conductance, minside volume, etc.
template <class G>
struct CutStats {
    // Ahh, dear C++
    using Node = typename G::Node;
    using Edge = typename G::Edge;
    using Cut = set<Node>;
    using EdgeIt = typename G::EdgeIt;
    using Bisection = set<Node>;

    size_t crossing_edges = 0;

private:
    // When CutStats is created, it gets a cut but it doesnt know whether it is the larger or smaller side, so it
    // finds out and then adapts the queries accordingly
    bool is_min_side; // Whether the side represented by the cut is the minimum side

    size_t min_side = 0;
    size_t cut_volume = 0;
    size_t max_side = 0;
    size_t cut_size = 0;
    size_t othersize = 0;
    size_t non_cut_volume = 0;

    // Could probably be refactored around
    long noncut_volume () { return non_cut_volume; }
public:
    // The passing of number of vertices actually kinda sucks and I would refactor it away if I had time
    CutStats(const G &g, size_t num_vertices, const Cut &cut) {
        initialize(g, num_vertices, cut);
    }

    void initialize(const G &g, size_t num_vertices, const Cut &cut) {
        for (EdgeIt e(g); e != INVALID; ++e) {
            if (is_crossing(g, cut, e)) crossing_edges += 1;
            if (cut.count(g.u(e))) cut_volume += 1;
            else non_cut_volume += 1;
            // If it's a self loop, it only contributes once, but if not, it contributes twice
            if (g.u(e) == g.v(e)) continue;
            if (cut.count(g.v(e))) cut_volume += 1;
            else non_cut_volume += 1;
        }

        assert(cut.size() <= num_vertices);
        size_t other_size = num_vertices - cut.size();
        min_side = min(cut.size(), other_size);
        max_side = max(cut.size(), other_size);
        is_min_side = cut.size() == min_side;
        cut_size = cut.size();
        othersize = other_size;
    }

    static bool is_crossing(const G &g, const Bisection &c, const Edge &e) {
        bool u_in = c.count(g.u(e));
        bool v_in = c.count(g.v(e));
        return u_in != v_in;
    }

    // Whether any of the endpoints of the edge are in this side of the cut
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

    long min_volume() {
        return min(cut_volume, non_cut_volume);
    }

    double conductance() {
        return min_side == 0 ? 999 : crossing_edges * 1. / min_volume();
    }

    // These printouts are actually what's used in the run.py analysis code. It's highly dependent on the exact form
    // here.
    void print(string prefix="") {
        l.progress() << prefix << "Edge crossings (E) : " << crossing_edges << endl;
        l.progress() << prefix << "cut size: ( " << cut_size << " | " << othersize << " )" << endl
             << "diff: " << diff() << " (factor " << imbalance() << " of total n vertices)" << endl;
        l.progress() << prefix << "cut volumes: ( " << cut_volume << " | " << non_cut_volume << " )" << endl;
        l.progress() << prefix << "expansion: " << expansion() << endl;
        l.progress() << prefix << "conductance: " << conductance() << endl;
    }
};
// Reads the file filename,
// creates that graph in graph g which is assumed to be empty
// In the process fills "nodes" with each node created at the index of "its id in the file minus one"
// And sets each node's original_ids id to be "its id in the file minus one".
// Of course original_ids must be initialized onto the graph g already earlier.
static void parse_chaco_format
        ( const string &filename
        , ListGraph &g
        , vector<Node> &nodes
        , bool take_multi_edges
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

    // We know how many nodes to create. Create them at once, so we can add edges to "future" nodes.
    for (size_t i = 0; i < n_verts; i++) {
        Node n = g.addNode();
        nodes.push_back(n);
    }

    // Now read all the lines and add edges
    for (size_t i = 0; i < n_verts; i++) {
        getline(file, line);
        istringstream iss(line); // This is one way to word-split in C++.
        vector<string> tokens{istream_iterator<string>{iss},
                              istream_iterator<string>{}};
        Node u = nodes[i]; // Current node we're working with
        // This will be very easy. All ID's on the line are edges
        for(string& str : tokens) {
            // This code could use some more assertions...
            size_t v_name = stoi(str);
            l.debug() << "edge from " << i << " to: " << v_name << "..." ;
            assert(v_name != 0);

            Node v = nodes[v_name - 1];

            if (take_multi_edges || findEdge(g, u, v)== INVALID) {
                g.addEdge(u, v);
            }
        }
    }

    cout << countEdges(g) << endl;
    cout << countNodes(g) << endl;
    assert(countEdges(g) == n_edges);
    assert(countNodes(g) == n_verts);
}

// This code facilitates in-program generation of graphs. Honestly, uh, don't use this, just generate your graphs
// externally with some python scripts and actually save them to a file. It will be way more dependable.
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

// Writes out a cut to a file. One line for each node, with "1" if it's on one side, "0" if the other.
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

// Format assumed to be same as write_cut above
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

// Brancher to make sure we have a graph to work from, wherever we're supposed to get it.
void initGraph(GraphContext &gc, InputConfiguration config) {
    if (config.load_from_file) {
        parse_chaco_format(config.file_name, gc.g, gc.nodes, !config.ignore_multi);
    } else {
        l.debug() << "Generating graph with " << config.n_nodes_to_generate << " nodes." << endl;
        generate_large_graph(gc.g, gc.nodes, config.n_nodes_to_generate);
    }

    // I would refactor num_edges away if I had time. Now, it is being used however, so can't just delete.
    gc.num_edges = countEdges(gc.g);
    l.debug() << "gc.num_edges: " << gc.num_edges << endl;
}

// For some reason lemon returns arbitrary values for flow. You only get the correct value by taking the difference.
// The arc lookup is necessary for performance. Don't let LEMON look for arcs with its own lookup methods, it's way
// too slow.
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

// Note that this prints graph ID's. That's not the same as their positions in "nodes", so can you trust this output?
// Just be careful not to make assumptions.
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

// This actually copies the graph.
void createSubdividedGraph(SubdividedGraphContext& sgc) {
    graphCopy(sgc.origContext.g, sgc.sub_g)
      .nodeRef(sgc.n_ref)
      .nodeCrossRef(sgc.n_cross_ref)
      .run();
    G& g = sgc.sub_g;
    for (NodeIt n(g); n != INVALID; ++n) {
        sgc.origs[n] = true;
    }
    // at this point it's just the pure copy, that's why we set them all to be "originals"

    // We'll need to iterate over these edges but we'll be deleting and poking in the graph.
    vector<Edge> edges;
    for (EdgeIt e(g); e != INVALID; ++e) {
        edges.push_back(e);
    }

    // Also, only the new verts added later should be in the only-splits adapter.
    for (NodeIt n(g); n != INVALID; ++n) {
        sgc.only_splits.disable(n);
    }

    // Do the subdivisioning! Replace every edge (u, v) with instead two new vertices (u, s) and (s, v) where s is a new
    // vertex, called a subdivision vertex.
    // This also applies if the edge is a multi-edge, each one is treated individually, and gets its own s created.
    // However, if the edge is a self-loop, we just replace it (u, u) with one new edge and vertex (s, u).
    // This is in accordance with Saranurak's notion of self-loops only contributing one volume.
    for(auto& e : edges) {
        Node u = g.u(e);
        Node v = g.v(e);
        g.erase(e);

        Node s = g.addNode();
        sgc.origs[s] = false;
        sgc.only_splits.enable(s);
        g.addEdge(u, s);

        // one-volume for selfloops notion.
        // Also, there is actually another reason why it's implemented this way. And that is: if you want to let the
        // subdivision graph be an actual multigraph, then the path decomposition code falls apart potentially or at
        // least needs serious review.
        if(u != v) g.addEdge(s, v);
        sgc.split_vertices.push_back(s);
    }

    // BIG OPTIMISATION OPPORTUNITY: Just create one new edge for each multi, and implement code to handle scaling the
    // capacities.
}

// Finally, the actual algorithm! All above are supporting structures.

// CUT MATCHING GAME IMPLEMENTATION
// Based off of
// "Graph partitioning using single commodity flows"
// by Khandekar, Vazirani, Rao
// https://dl.acm.org/doi/10.1145/1538902.1538903
// and
// "Expander Decomposition and Pruning: Faster, Stronger, and Simpler"
// by Saranurak and Wang.
// https://arxiv.org/abs/1812.08958

// The Cut-Matching game approximates sparsest cuts in graphs or certifies them as expanders.
// The two players take turns playing in rounds.
// THe Cut player produces bisections
// High-level Pseudocode:
//   Input: Graph G = (V, E), g_phi, h_phi
//   Matchings := []
//   Cuts := []
//   G’ = (Xe Union V, E’) := SubdivisionGraph(G)
//   While not ShouldStop():
//    Bisection := CutPlayer(Xe, Matchings)
//    (Matching, CutG’) := MatchingPlayer(G’, Bisection)
//    Cut := CutInG(CutG’)
//    Push(Matchings, Matching); Push(Cuts, Cut)
//   Return C in Cuts with minimal Conductance(C)
//
//   CutPlayer(V, Matchings):
//    Assign each vertex a random charge from {-1, 1}
//    For M in Matchings:
//     For (u, v) in M:
//      charge(u), charge(v) := avg(charge(u), charge(v))
//    Sort V by charge
//    Return (First half, second half) of V
//
//   MatchingPlayer(Graph, (S, S’)):
//    Create new vertices s, t; connect s with all of S, and all of S’ with t
//    LastCut := null
//    For C in [1, 2, 4, 8, …]:
//     Set all edge capacities to C
//     Flow := (Try to route n/2 flow from s to t with a MaxFlow algorithm)
//     If successful:
//      Matching :=DecomposePaths(Flow)
//      return (Matching, LastCut)
//     Else:
//      LastCut = MinCut(Flow)

struct CutMatching {
    const Configuration &config;
    GraphContext &gc;
    SubdividedGraphContext sgc;
    default_random_engine &random_engine;
    // Sub prefix just comes from the subdivision refactoring, could be removed
    vector<unique_ptr<RoundReport>> sub_past_rounds; // We just want the reports on the heap, not the stack
    vector<Matchingp> sub_matchings;
    bool reached_H_target = false;
    std::chrono::time_point<std::chrono::system_clock> start_time;
    bool timed_out = false;
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
    // inside a "matching context" that exists for the duration of the matching step
    struct MatchingContext {
        // This NEEDS to be the whole graph - not some adapter (but why?)
        G& g;
        Node s;
        Node t;
        EdgeMapi capacity;
        CutMap cut_map;
        Snapshot snap; //RAII - in Destuctor it reverts the state of the graph, at least as far as new
        // vertices and edges are concerned (seems this is a bit limited if you want more - consult LEMON). So this
        // gives us free cleanup after the matching step.

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

        // Extracts a copy of the cut, for conveniently putting in the roundReports I think
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

    // This fast decomposition scheme was the last step of optimizations until flow became the only bottleneck.
    // We mostly achieve speed by precomputing lookups of arcs and flow children. Directly querying LEMON most often
    // involves some stupid linear algorithm.
    // In any case, this function starts from u_orig, extracts one path to t through which there is still flow,
    // using flow_children, and returns the path (actually just start and end) as out_path. It also decreases the flows
    // and deletes from flow_children appropriately.
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
        while (true) { // This code is optimized and easy to get wrong. Just a watning.
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

    // Decompose the flow from a maxflow algorithm into a set of start/end pairs for the paths.
    // This makes sense in this context, since of course the flow goes into each node on one side of the bisection with
    // capacity one, so the only logical decomposition is that each node has a path to some other one and we get a
    // matching. This matching is not necessarily unique.
    // Currently this is a greedy one-path-after-another algorithm, one could maybe optimize it even more to e.g. take
    // advantage of large-flow-value edges but this is not a bottleneck right now so it's not necessary.
    // This algo starts with precomputing the flow children for speed, and the arc lookups as well.
    void decompose_paths_fast(const MatchingContext &mg, const unique_ptr<FlowAlgo> &f, Paths &out_paths) {
        f->startSecondPhase();
        EdgeMapi subtr(mg.g, 0); // Might not be in use anymore, was a solution for updating flow.
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

            out_paths.push_back(array<Node, 2>());
            extract_path_fast(mg.g, f, flow_children, u, mg.t, out_paths[out_paths.size() - 1]);
        }
    }

    // Just executes the maxflow/mincut algorithm, measures its time and prints it.
    void run_min_cut(const MatchingContext &mg, unique_ptr<FlowAlgo> &p) const {
        p.reset(new Preflow<G, EdgeMapi>(mg.g, mg.capacity, mg.s, mg.t));
        auto start2 = now();
        p->runMinCut(); // Note that "startSecondPhase" must be run to get flows for individual verts
        auto stop2 = now();
        l.progress() << "flow: " << p->flowValue() << " (" << duration_sec(start2, stop2) << " s)" << endl;

        if(config.timeout_after_minutes > 0 && duration_sec(start_time, now())/60.0 > config.timeout_after_minutes) {
          // We're over time and need to break
          throw JaniukException();
        }
    }

    // Small subroutine to set all capacities to cap, except the edges that touch source or sink (taken from mg).
    void set_matching_capacities(MatchingContext &mg, size_t cap) const {
        for (EdgeIt e(mg.g); e != INVALID; ++e) {
            if (mg.touches_source_or_sink(e)) continue;
            mg.capacity[e] = cap;
        }
    }

    // Try exponentially larger capacities inside the graph, until flow_target flow can be achieved.
    // NOTE: Returns result flow from the last capacity that could NOT accomodate flow_target flow. This is why the
    // logic in the loop can seem a bit strange.
    MatchResult bin_search_flows(MatchingContext &mg, unique_ptr<FlowAlgo> &p, size_t flow_target) const {
        auto start = now();
        size_t cap = 1; // Capacity
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
        // If the for loop is exited without reaching full flow, then that's a severe issue with the graph itself
        // and basically, a bug. Might be something like the graph is not connected.

        // Not we copy out the cut
        MatchResult result{mg.extract_cut(), cap};

        auto stop = now();
        l.progress() << "Flow search took (seconds) " << duration_sec(start, stop) << endl;

        return result;
    }

    void decompose_paths(const MatchingContext &mg, const unique_ptr<FlowAlgo> &p, vector<array<Node, 2>> &paths) {
        decompose_paths_fast(mg, p, paths);
    }

    // Create sink and source vertices, and attach them to the two sides of the cut
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

    // Cut player routine of creating a bisection of the split-nodes in the subdivision graph.
    // Returns the bisection. Uses the list of previous matchings.
    // Also returns, through h_multi_cond_out, the conductance of the returned bisection.
    template<typename GG, typename M>
    Bisectionp cut_player(const GG &g, const vector<unique_ptr<M>> &given_matchings, double &h_multi_cond_out) {
        l.debug() << "Running Cut player" << endl;
        typename GG::template NodeMap<double> probs(g); // Probabilities, aka Charges
        vector<Node> all_nodes;

        // assign all nodes a random (+-)1/N probability

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

        // Now sort on charge.
        // I'm shuffling before sorting to avoid dependency on original order; sort is probably stable and we have many
        // equalities
        shuffle(all_nodes.begin(), all_nodes.end(), random_engine);
        sort(all_nodes.begin(), all_nodes.end(), [&](Node a, Node b) { return probs[a] < probs[b]; });

        // aand cut the list in half
        size_t size = all_nodes.size();
        all_nodes.resize(size / 2);
        auto b = Bisectionp(new Bisection(all_nodes.begin(), all_nodes.end()));
        l.debug() << "Cut player gave the following cut: " << endl;
        print_cut(*b, l.debug());

        // Yeah, we're printing for both variants of H since it's still unclear which is the relevant one.
        // Research question for the future: Should H's conductance be computed with multiedges in mind?
        auto hcs = CutStats<decltype(H)>(H, num_vertices, *b);
        l.progress() << "H expansion: " << hcs.expansion() << ", num cross: " << hcs.crossing_edges << endl;
        l.progress() << "H conductance: " << hcs.conductance() << ", num cross: " << hcs.crossing_edges << endl;
        h_multi_cond_out = hcs.conductance();

        auto hscs = CutStats<decltype(H_single)>(H_single, num_vertices, *b);
        l.progress() << "H_single expansion: " << hscs.expansion() << ", num cross: " << hscs.crossing_edges << endl;
        l.progress() << "H_single conductance: " << hscs.conductance() << ", num cross: " << hscs.crossing_edges << endl;

        return b;
    }

    // Matching player. Runs flow through the bisection and returns cut.
    MatchResult sub_matching_player(const set<Node> &bisection, vector<array<Node, 2>>& m_out) {
        MatchingContext mg(sgc.sub_g);
        make_sink_source(sgc.only_splits, mg, bisection);

        // These two lines probably pointless but if I delete them I have to run all my tests again.
        Node s = mg.s;
        int id = sgc.sub_g.id(s);

        unique_ptr<FlowAlgo> p;
        MatchResult mr = bin_search_flows(mg, p, sgc.split_vertices.size()/2);

        decompose_paths(mg, p, m_out);
        return mr;
    }

    // The volume mechanism was added to prevent happily returning a cut with good conductance but balance below a
    // certain treshold, and so this is the treshold for the volume of the smaller side.
    long sub_volume_treshold() {
        return config.volume_treshold_factor * sgc.origContext.nodes.size();
    }

    // Runs one round of the whole algorithm
    unique_ptr<RoundReport> sub_one_round() {
      // Results will be written into the round_report
      unique_ptr<RoundReport> report = make_unique<RoundReport>();

      // First, the cut player "plays" and creates a bisection
      Bisectionp sub_bisection = cut_player(sgc.only_splits, sub_matchings, report->multi_h_conductance);

      // Then the matching player "plays" and creates a matching
      Matchingp smatchingp(new Matching());
      MatchResult smr = sub_matching_player(*sub_bisection, *smatchingp);

      // We add the matching to the list of previous ones
      sub_matchings.push_back(move(smatchingp));

      // =======================
      // The rest of this function is bookkeeping
      // =======================

      // Write down relevant stuff into the report
      report->index = sub_past_rounds.size();
      report->capacity_required_for_full_flow = smr.capacity;

      report->cut = make_unique<Cut>();
      for(auto& n : *(smr.cut_from_flow)) {
        if(sgc.origs[n]) {
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

    // Figures out from all the bookkeeping if we should stop.
    // Depending on configuration options there may be several reasons to stop, e.g. #rounds, cut conductnce, H conductance...
    // There's also some logic on volume. You could basically just ignore that, it was added as a feature for use in the
    // larger algorithm (Saranurak) but there were bigger problems it seemed.
    bool sub_should_stop() {
        int i = sub_past_rounds.size();
        if(i == 0) return false;
        if(i >= config.max_rounds && config.max_rounds != 0) {
          l.progress() << "Max rounds reached, should stop." << endl;
          return true;
        }

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
        return false;
    }

    // Loop that runs the whole algorithm. Prints a nice message after every round
    void run() {
      // bin_search_flows will compare agains this and throw an exception if we are over time.
      // Because that's the most time consuming operation. It's an ugly solution but kinda should work.
      start_time = now();


      int num_rounds_completed = 0;
      try {
        while (!sub_should_stop()) {
          sub_past_rounds.push_back(sub_one_round());
          num_rounds_completed++;
          print_end_round_message(sub_past_rounds.size()-1);
        }
      } catch(JaniukException& e) { // Poor man's timer interrupt. With more time I would have redesigned this...
        l.progress() << "We ran out of time, wrapping up!" << endl;
        timed_out = true;
      }

      if(sub_past_rounds.size() > num_rounds_completed) {
        sub_past_rounds.pop_back();
      }
      assert(num_rounds_completed == sub_past_rounds.size());
    }
};

// All the command-line options are created here, but parsed in parse_options. So if parse_options forgets about one of
// these, it will be a noop.
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
      ("timeout_m", "After this many minutes, try to break the algorithm and just report on the rounds until now. Might carry a small delay.",
        cxxopts::value<float>())
            ;
    return options;
}

// Go through the command line options that were given, update code variables to make the changes happen.
// Mosly writes to the centralized "config" object.
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

    if(result.count("timeout_m")) {
      config.timeout_after_minutes = result["timeout_m"].as<float>();
    }
}

int main(int argc, char **argv) {
    Configuration config;
    parse_options(argc, argv, config);
    // And from here, all configuration should already be set up.

    if(config.show_help_and_exit) return 0;

    GraphContext gc;
    initGraph(gc, config.input);
    // Now the input graph is loaded (or possibly generated).

    // Just initalize the random engine.
    default_random_engine random_engine = config.seed_randomness
                    ? default_random_engine(config.seed)
                    : default_random_engine(random_device()());

    // And run the whole algorithm!
    CutMatching cm(gc, config, random_engine);
    cm.run();

    // We should have run some rounds haha
    assert(!cm.sub_past_rounds.empty());

    // Best by conductance
    auto& best_round = *min_element(cm.sub_past_rounds.begin(), cm.sub_past_rounds.end(), [](auto &a, auto &b) {
        return a->g_conductance < b->g_conductance;
    });


    // The rest is printing output


    // Quick summaries of all the rounds
    for(int i = 0; i < cm.sub_past_rounds.size(); i++) {
        l.progress() << "R" << i << " cond " << cm.sub_past_rounds[i]->g_conductance << endl;
    }

    // This string lies, it's conductance that counts now
    l.progress() << "The best with best expansion was found on round" << best_round->index << endl;
    auto &best_cut = best_round->cut;
    CutStats<G>(gc.g, gc.nodes.size(), *best_cut).print("final_");

    if(cm.timed_out) {
      l.progress() << "CASE4 Time ran out, we were not able to finish the algorithm." << endl;
    } else if(config.use_H_phi_target && config.use_G_phi_target && config.use_volume_treshold) {
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

    // As a service, read and analyze the walshaw partition already given.
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
