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

// TODO now
// Clean the code much more, to be able to do the stopping and edge version
// Basically get unnecessary stuff out of the algo

// TODO OK how do we do edge version...
// Whats the plan for hte edge version? Lets describe it in text
// 1. At the start of the algo, we need to make the subdivision graph.
// 2. And we need the set of subivided vertices. we could store it in the context to start with.
// 3. Then cut player needs to do the cut on those instead, can this be cone in an opaque way?
// 4. The matching player has to push flow differently and compile the cut differently. This will be a big difference.


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
template <typename GG>
using Matching = ListEdgeSet<GG>;
template <typename GG>
using Matchingp = unique_ptr<Matching<GG>>;
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
};

// I'd say implementing our own adaptor is more effort, we can just do the snapshot thing
// Actually lets just subdivide manually at the start and we dont even need to restore.
struct SubdividedGraphContext {
    SubdividedGraphContext (GraphContext& gc) :
    origContext(gc),
    nf(sub_g),
    ef(sub_g),
    only_splits(sub_g, nf, ef) {} ;
    GraphContext& origContext;
    G sub_g;
    NodeMapb nf;
    EdgeMapb ef;
    SubGraph<G> only_splits;
    vector<Node> split_vertices;
};

// TODO What chnages will be necessary?
struct RoundReport {
    size_t index;
    size_t capacity_required_for_full_flow;
    double multi_h_expansion;
    double g_expansion;
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
    size_t num_edges = 0;
    long degreesum() { return num_edges*2;}
    long noncut_volume () { return degreesum() - cut_volume;}

public:
    CutStats(const G &g, size_t num_vertices, const Cut &cut) {
        initialize(g, num_vertices, cut);
    }

    void initialize(const G &g, size_t num_vertices, const Cut &cut) {
        for (EdgeIt e(g); e != INVALID; ++e) {
            ++num_edges;
            if (is_crossing(g, cut, e)) crossing_edges += 1;
            if (any_in_cut(g, cut, e)) cut_volume += 1;
        }

        assert(cut.size() <= num_vertices);
        size_t other_size = num_vertices - cut.size();
        min_side = min(cut.size(), other_size);
        max_side = max(cut.size(), other_size);
        is_min_side = cut.size() == min_side;
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

    void print() {
        cout << "Edge crossings (E) : " << crossing_edges << endl;
        cout << "cut size: (" << min_side << " | " << max_side << ")" << endl
             << "diff: " << diff() << " (" << imbalance() << " of total n vertices)" << endl;
        cout << "Min side: " << min_side << endl;
        cout << "E/min(|S|, |comp(S)|) = " << expansion() << endl;
    }
};
// Reads the file filename,
// creates that graph in graph g which is assumed to be empty
// In the process fills nodes with each node created at the index of (its id in the file minus one)
// And sets each node's original_ids id to be (its id in the file minus one).
// Of course original_ids must be initialized onto the graph g already earlier.
static void parse_chaco_format(const string &filename, ListGraph &g, vector<Node> &nodes) {
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
    g.reserveNode(n_verts);
    g.reserveNode(n_edges);

    for (size_t i = 0; i < n_verts; i++) {
        Node n = g.addNode();
        nodes.push_back(n);
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
        l.progress() << "Odd number of vertices, adding extra one." << endl;
        Node n = g.addNode();
        g.addEdge(nodes[0], n);
        nodes.push_back(n);
    }
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
        cout << "Cannot open file " << file_name << endl;
        return;
    }

    cout << "Writing partition with "
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
        parse_chaco_format(config.file_name, gc.g, gc.nodes);

    } else {
        l.debug() << "Generating graph with " << config.n_nodes_to_generate << " nodes." << endl;
        generate_large_graph(gc.g, gc.nodes, config.n_nodes_to_generate);
    }

    gc.num_edges = countEdges(gc.g);
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

template <typename GG>
void print_matching(const Matchingp<GG> &m, decltype(cout)& stream) {
    for (typename Matching<GG>::EdgeIt e(*m); e != INVALID; ++e) {
        stream << "(" << m->id(m->u(e)) << ", " << m->id(m->v(e)) << "), ";
    }
    stream << endl;
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
    graphCopy(sgc.origContext.g, sgc.sub_g).run();
    G& g = sgc.sub_g;

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
        g.erase(e);

        Node s = g.addNode();
        sgc.only_splits.enable(s);
        g.addEdge(u, s);
        g.addEdge(s, v);

        sgc.split_vertices.push_back(s);
    }
}

struct CutMatching {
    const Configuration &config;
    GraphContext &gc;
    SubdividedGraphContext sgc;
    default_random_engine &random_engine;
    vector<unique_ptr<RoundReport>> past_rounds;
    vector<Matchingp<G>> matchings;
    vector<Matchingp<SubGraph<G>>> sub_matchings;
    bool reached_H_target = false;
    // Input graph
    CutMatching(GraphContext &gc, const Configuration &config_, default_random_engine &random_engine_)
    :
    config(config_),
    gc(gc),
    sgc{gc},
    random_engine(random_engine_)
    {
        assert(gc.nodes.size() % 2 == 0);
        assert(gc.nodes.size() > 0);
        assert(connected(gc.g));

        createSubdividedGraph(sgc);
    };

    // During the matching step a lot of local setup is actually made, so it makes sense to group it
    // inside a "matching context" that exists for the duration of the mathing step
    struct MatchingContext {
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
            auto &tup = flow_children[u].back();
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

    void runMinCut(const MatchingContext &mg, unique_ptr<FlowAlgo> &p) const {
        p.reset(new Preflow<G, EdgeMapi>(mg.g, mg.capacity, mg.s, mg.t));
        auto start2 = now();
        p->runMinCut(); // Note that "startSecondPhase" must be run to get flows for individual verts
        auto stop2 = now();
        l.progress() << "flow: " << p->flowValue() << " (" << duration_sec(start2, stop2) << " s)" << endl;
    }

    void setMatchingCapacities(MatchingContext &mg, size_t cap) const {
        for (EdgeIt e(mg.g); e != INVALID; ++e) {
            if (mg.touches_source_or_sink(e)) continue;
            mg.capacity[e] = cap;
        }
    }

    MatchResult bin_search_flows(MatchingContext &mg, unique_ptr<FlowAlgo> &p, size_t flow_target) const {
        auto start = now();
        size_t cap = 1;
        //for (; cap < mg.gc.nodes.size(); cap *= 2) {
        for (; cap < flow_target*2; cap *= 2) {
            l.progress() << "Cap " << cap << " ... " << flush;
            setMatchingCapacities(mg, cap);
            runMinCut(mg, p);

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



    // Actually, cut player gets H
// Actually Actually, sure it gets H but it just needs the matchings...
// TODO Ok so can we just call this with split_only and matchings of those?
    template<typename GG, typename M>
    Bisectionp cut_player(const GG &g, const vector<unique_ptr<M>> &given_matchings, double &h_multi_exp_out) {
        l.debug() << "Running Cut player" << endl;
        using MEdgeIt = typename M::EdgeIt;

        typename GG::template NodeMap<double> probs(g);
        vector<Node> all_nodes;

        uniform_int_distribution<int> uniform_dist(0, 1);
        for (typename GG::NodeIt n(g); n != INVALID; ++n) {
            all_nodes.push_back(n);
            probs[n] = uniform_dist(random_engine)
                       ? 1.0 / all_nodes.size()
                       : -1.0 / all_nodes.size();
        }

        size_t num_vertices = all_nodes.size();

        ListEdgeSet H(g);
        ListEdgeSet H_single(g);
        for (const unique_ptr<M> &m : given_matchings) {
            for (MEdgeIt e(*m); e != INVALID; ++e) {
                Node u = m->u(e);
                Node v = m->v(e);
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
        h_multi_exp_out = hcs.expansion();
        auto hscs = CutStats<decltype(H_single)>(H_single, num_vertices, *b);
        l.progress() << "H_single expansion: " << hscs.expansion() << ", num cross: " << hscs.crossing_edges << endl;

        return b;
    }

    // returns capacity that was required
// Maybe: make the binsearch an actual binsearch
// TODO Let listedgeset just be 2-arrays of nodes. Lemon is getting in the way too much.
// But also what is assigned in MtchResult?
    MatchResult matching_player(const set<Node> &bisection, ListEdgeSet<G> &m_out) {
        MatchingContext mg(gc.g);
        make_sink_source(mg, bisection);

        unique_ptr<FlowAlgo> p;
        MatchResult mr = bin_search_flows(mg, p, gc.nodes.size()/2);

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

        return mr;
    }
/*
    // returns capacity that was required
// Maybe: make the binsearch an actual binsearch
// TODO Let listedgeset just be 2-arrays of nodes. Lemon is getting in the way too much.
// But also what is assigned in MtchResult?
    MatchResult sub_matching_player(const set<Node> &bisection, vector<array<Node, 2>>& m_out) {
        MatchingContext mg(gc);
        make_sink_source(mg, bisection);

        unique_ptr<FlowAlgo> p;
        MatchResult mr = bin_search_flows(mg, p);

        decompose_paths(mg, p, m_out);

        // Now how do we extract the cut?
        // In this version, in one run of the matching the cut is strictly decided. We just need
        // to decide which one of them.
        // Only when we change to edge will the cut need to be explicitly extracted.
        // Rn the important thing is to save cuts between rounds so I can choose the best.

        return mr;
    }

*/
    long volume_treshold() {
        return config.volume_treshold_factor * gc.num_edges;
    }

    // Ok lets attack from here
    // Theres a lot of risk for problems with "is this a cut in the orig graph or in the splits?
    unique_ptr<RoundReport> one_round() {
        unique_ptr<RoundReport> report = make_unique<RoundReport>();

        // So this needs to be a bisection of splitnodes, I feel this could be very opaque.
        // We'd need a subgraph that is just the splitnodes
        Bisectionp bisection = cut_player(gc.g, matchings, report->multi_h_expansion);

        // WIP SUB
        double h_multi_out_sub = 0;
        Bisectionp sub_bisection = cut_player(sgc.only_splits, sub_matchings, h_multi_out_sub);
        // Well ok, it's doing the first random thing well.
        // TODO test on rest...


        // Matching on splitnodes, but we need to also save the actual cut...
        Matchingp<G> matchingp(new Matching<G>(gc.g));
        MatchResult mr = matching_player(*bisection, *matchingp);
        matchings.push_back(move(matchingp));

        // WIP SUB
        Matchingp<SubGraph<G>> smatchingp(new Matching<SubGraph<G>>(sgc.only_splits));
        //MatchResult mr = matching_player(*sub_bisection, *smatchingp);
        //sub_matchings.push_back(move(matchingp));

        //c.cuts.push_back(move(mr.cut_from_flow));
        report->index = past_rounds.size();
        report->capacity_required_for_full_flow = mr.capacity;
        report->cut = move(mr.cut_from_flow);
        auto cs = CutStats<G>(gc.g, gc.nodes.size(), *report->cut);
        report->g_expansion = cs.expansion();
        l.progress() << "G cut expansion " << report->g_expansion << endl;
        report->volume = cs.minside_volume();
        l.progress() << "G cut minside volume " << cs.minside_volume() << endl;
        l.progress() << "G cut maxside volume " << cs.maxside_volume() << endl;
        report->relatively_balanced = report->volume > volume_treshold();
        return move(report);

        // We want to implement that it parses partitions
        // That has nothing to do with the rounds lol
    }


    // Stopping condition
    bool should_stop() {
        int i = past_rounds.size();
        if(i == 0) return false;
        if(i >= config.max_rounds && config.max_rounds != 0) return true;

        const auto& last_round = past_rounds[past_rounds.size() - 1];
        if(config.use_H_phi_target && last_round->multi_h_expansion >= config.H_phi_target) {
            cout << "H Expansion target reached, this will be case 1 or 3. According to theory, this means we probably won't find a better cut. That is, assuming you set H_phi right. "
                    "If was used together with G_phi target, this also certifies the input graph is a G_phi expander unless there was a very unbaanced cut somewhere, which we will proceed to look for." << endl;
            reached_H_target = true;
            return true;
        }

        if(config.use_G_phi_target)
        if(last_round->g_expansion >= config.G_phi_target) {
            if(config.use_volume_treshold && last_round->relatively_balanced) {
                cout << "CASE2 G Expansion target reached with a cut that is relatively balanced. Cut-matching game has found a balanced cut as good as you wanted it."
                     << endl;
                return true;
            }

            if(!config.use_volume_treshold) {
                cout << "G Expansion target reached. Cut-matching game has found a cut as good as you wanted it. Whether it is balanced or not is up to you."
                     << endl;
                return true;
            }
        }
    }

    void run() {
        while (!should_stop()) {
            past_rounds.push_back(one_round());
            print_end_round_message(past_rounds.size()-1);
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

    CutMatching cm(gc, config, random_engine);
    cm.run();

    assert(!cm.past_rounds.empty());
    // Best by expnansion
    auto& best_round = *max_element(cm.past_rounds.begin(), cm.past_rounds.end(), [](auto &a, auto &b) {
        return a->g_expansion < b->g_expansion;
    });

    cout << "The best with highest expansion was found on round" << best_round->index << endl;
    cout << "Best cut sparsity: " << endl;
    auto &best_cut = best_round->cut;
    CutStats<G>(gc.g, gc.nodes.size(), *best_cut).print();

    if(config.use_H_phi_target && config.use_G_phi_target && config.use_volume_treshold) {
        if(cm.reached_H_target) {
            if(best_round->g_expansion < config.G_phi_target) {
                cout << "CASE1 NO Goodenough cut, G certified expander." << endl;
            } else {
                cout << "CASE3 Found goodenough but very unbalanced cut." << endl;
            }
        } else {
            cout << "CASE2 Goodenough balanced cut" << endl;
        }

    }

    if (config.output_cut) { write_cut(gc.nodes, *best_cut, config.output_file); }

    if (config.compare_partition) {
        Cut reference_cut;
        read_partition_file(config.partition_file, gc.nodes, reference_cut);

        cout << endl
             << "The given partition achieved the following:"
             << endl;
        CutStats<G>(gc.g, gc.nodes.size(), reference_cut).print();
    }

    return 0;
}

#pragma clang diagnostic pop
