import numpy as np
import networkx as nx
from collections import defaultdict
import networkx as nx
import os
import cvxpy as cp
import sqlparse
import pickle
from networkx.readwrite import json_graph


info_width = 300
p1_width = 1000
p1_height = 600

SOURCE_NODE = tuple("s")
NILJ_CONSTANT = 0.001
MAX_JOINS = 32

COST_MODEL = "C"
COST_KEY =  "est_cost"
TRUE_COST_KEY = "true_cost"

# cost_key
# QUERY_DIR = "./debug_sqls/"
QUERY_DIR = "./queries/"

# query = "1.pkl"
# QUERY_DIR = "./queries/imdb/1a/"
# # good ones
# query = "1a40.pkl"
# query = "1a200.pkl"

# QUERY_DIR = "./queries/imdb/8a/"
# # good ones
# query = "8a40.pkl"

MIN_RADIUS = 20.0
MAX_RADIUS = 50.0
NODE_WIDTH = 3.0

def get_node_sizes(ests, min_size, max_size):
    new_range = max_size-min_size
    old_range = max(ests) - min(ests)
    sizes = []
    for ni,est in enumerate(ests):
        #NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
        size = (((ests[ni] - min(ests)) * new_range) / old_range) + min_size
        sizes.append(size)

    return sizes


def get_flows(subsetg, cost_key):
    # TODO: add options for changing beta; look at old code
    # FIXME: assuming subsetg  is S->D; construct_lp expects it to be D->S.

    edges, costs, A, b, G, h = construct_lp(subsetg, cost_key=cost_key)
    return np.zeros(len(subsetg.edges)), edges
    if len(subsetg.edges) > 5000:
        return np.zeros(len(subsetg.edges)), edges

    n = len(edges)
    P = np.zeros((len(edges),len(edges)))
    for i,c in enumerate(costs):
        P[i,i] = c

    q = np.zeros(len(edges))
    x = cp.Variable(n)
    #prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
    #                 [G @ x <= h,
    #                  A @ x == b])
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                     [A @ x == b])
    prob.solve(verbose=False, solver=cp.OSQP)
    flows = np.array(x.value)

    return flows, edges


def construct_lp(subsetg, cost_key="cost"):
    '''
    @ret:
        list of node names
        node_names : idx
        edge_names : idx for the LP
        A: |V| x |E| matrix
        b: |V| matrix
        where the edges
    '''
    node_dict = {}
    edge_dict = {}
    b = np.zeros(len(subsetg.nodes()))
    A = np.zeros((len(subsetg.nodes()), len(subsetg.edges())))

    nodes = list(subsetg.nodes())
    nodes.sort()

    # node with all tables is source, node with no tables is target
    source_node = nodes[0]
    for i, node in enumerate(nodes):
        node_dict[node] = i
        if len(node) > len(source_node):
            source_node = node
    target_node = tuple("s")
    b[node_dict[source_node]] = 1
    b[node_dict[target_node]] = -1

    edges = list(subsetg.edges())
    edges.sort()
    for i, edge in enumerate(edges):
        edge_dict[edge] = i

    for ni, node in enumerate(nodes):
        in_edges = subsetg.in_edges(node)
        out_edges = subsetg.out_edges(node)
        for edge in in_edges:
            idx = edge_dict[edge]
            assert A[ni,idx] == 0.00
            A[ni,idx] = -1

        for edge in out_edges:
            idx = edge_dict[edge]
            assert A[ni,idx] == 0.00
            A[ni,idx] = 1

    G = np.eye(len(edges))
    G = -G
    h = np.zeros(len(edges))
    c = np.zeros(len(edges))
    # find cost of each edge
    for i, edge in enumerate(edges):
        c[i] = subsetg[edge[0]][edge[1]][cost_key]

    return edges, c, A, b, G, h

def load_sql_rep(fn):
    assert ".pkl" in fn
    try:
        with open(fn, "rb") as f:
            query = pickle.load(f)
    except Exception as e:
        print(e)
        print(fn + " failed to load...")
        exit(-1)

    query["subset_graph"] = \
            nx.OrderedDiGraph(json_graph.adjacency_graph(query["subset_graph"]))
    query["join_graph"] = json_graph.adjacency_graph(query["join_graph"])
    if "subset_graph_paths" in query:
        query["subset_graph_paths"] = \
                nx.OrderedDiGraph(json_graph.adjacency_graph(query["subset_graph_paths"]))

    return query

def compute_costs(subset_graph, cost_model,
        cost_key="cost", ests=None):
    '''
    @computes costs based on a simple cost model.
    '''
    total_cost = 0.0
    cost_key = cost_model + cost_key

    for edge in subset_graph.edges():
        if len(edge[0]) == len(edge[1]):
            # special case: source node --> single table node edges
            subset_graph[edge[0]][edge[1]][cost_key] = 1.0
            continue

        if len(edge[1]) > len(edge[0]):
            print(edge)
            pdb.set_trace()

        assert len(edge[1]) < len(edge[0])

        node1 = edge[1]
        diff = set(edge[0]) - set(edge[1])
        node2 = list(diff)
        node2.sort()
        node2 = tuple(node2)
        assert node2 in subset_graph.nodes()
        # joined node
        node3 = edge[0]
        cards1 = subset_graph.nodes()[node1]["cardinality"]
        cards2 = subset_graph.nodes()[node2]["cardinality"]
        cards3 = subset_graph.nodes()[edge[0]]["cardinality"]

        if isinstance(ests, str):
            # FIXME:
            if node1 == SOURCE_NODE:
                card1 = 1.0
            else:
                card1 = cards1[ests]

            if node2 == SOURCE_NODE:
                card2 = 1.0
            else:
                card2 = cards2[ests]
            card3 = cards3[ests]

        elif ests is None:
            # true costs
            card1 = cards1["actual"]
            card2 = cards2["actual"]
            card3 = cards3["actual"]
        else:
            assert isinstance(ests, dict)
            if node1 in ests:
                card1 = ests[node1]
                card2 = ests[node2]
                card3 = ests[node3]
            else:
                card1 = ests[" ".join(node1)]
                card2 = ests[" ".join(node2)]
                card3 = ests[" ".join(node3)]

        cost, edges_kind = get_costs(subset_graph, card1, card2, card3, node1,
                node2, cost_model)
        assert cost != 0.0
        subset_graph[edge[0]][edge[1]][cost_key] = cost
        subset_graph[edge[0]][edge[1]][cost_key + "scan_type"] = edges_kind

        total_cost += cost
    return total_cost

def get_costs(subset_graph, card1, card2, card3, node1, node2,
        cost_model):
    '''
    '''
    def update_edges_kind_with_seq(edges_kind, nilj_cost, cost2):
        if cost2 is not None and cost2 < nilj_cost:
            cost = cost2
            if len(node1) == 1:
                edges_kind["".join(node1)] = "Seq Scan"
            if len(node2) == 1:
                edges_kind["".join(node2)] = "Seq Scan"
        else:
            cost = nilj_cost
            if len(node1) == 1:
                edges_kind["".join(node1)] = "Index Scan"
                if len(node2) == 1:
                    edges_kind["".join(node2)] = "Seq Scan"
            elif len(node2) == 1:
                edges_kind["".join(node2)] = "Index Scan"
                if len(node1) == 1:
                    edges_kind["".join(node1)] = "Seq Scan"

    edges_kind = {}
    if cost_model == "C":
        # simple cost model for a left deep join
        assert len(node2) == 1, "node 2 must be a single table"
        nilj_cost = card1 + NILJ_CONSTANT*card2
        cost2 = card1*card2

        if cost2 < nilj_cost:
            cost = cost2
        else:
            cost = nilj_cost
        update_edges_kind_with_seq(edges_kind, nilj_cost, cost2)
    else:
        assert False, "cost model {} unknown".format(cost_model)

    return cost, edges_kind

def add_single_node_edges(subset_graph, source=None):
    global SOURCE_NODE
    if source is None:
        source = tuple("s")
    else:
        SOURCE_NODE = source

    # source = SOURCE_NODE
    # print(SOURCE_NODE)

    subset_graph.add_node(source)
    subset_graph.nodes()[source]["cardinality"] = {}
    subset_graph.nodes()[source]["cardinality"]["actual"] = 1.0

    for node in subset_graph.nodes():
        if len(node) != 1:
            continue
        if node[0] == source[0]:
            continue

        # print("going to add edge from source to node: ", node)
        subset_graph.add_edge(node, source, cost=0.0)
        in_edges = subset_graph.in_edges(node)
        out_edges = subset_graph.out_edges(node)
        # print("in edges: ", in_edges)
        # print("out edges: ", out_edges)

        # if we need to add edges between single table nodes and rest
        for node2 in subset_graph.nodes():
            if len(node2) != 2:
                continue
            if node[0] in node2:
                subset_graph.add_edge(node2, node)

def update_costs(subsetg, cost_model, y, cost_key):
    add_single_node_edges(subsetg, SOURCE_NODE)

    _ = compute_costs(subsetg, cost_model,
                                 cost_key=cost_key,
                                 ests=y)

    flows, edges = get_flows(subsetg, cost_model+cost_key)

    edge_widths = {}
    for i, x in enumerate(flows):
        subsetg.edges()[edges[i]][cost_key + "flow"] = x

    subsetg.remove_node(SOURCE_NODE)

def get_shortest_path(subsetg, cost_model, y, cost_key):
    update_costs(subsetg, cost_model, y, cost_key)
    add_single_node_edges(subsetg, SOURCE_NODE)

    nodes = list(subsetg.nodes())
    nodes.sort(key=lambda x: len(x))
    final_node = nodes[-1]

    ## FIXME: why do we always need this
    subsetg = subsetg.reverse()
    opt_labels_list = nx.shortest_path(subsetg, SOURCE_NODE,
            final_node, weight=cost_model+cost_key)

    for node in subsetg.nodes():
        if node in opt_labels_list:
            subsetg.nodes()[node][cost_model+cost_key+"-shortest_path"] = 1
        else:
            subsetg.nodes()[node][cost_model+cost_key+"-shortest_path"] = 0
    # TODO: store this as a node property

    subsetg = subsetg.reverse()
    subsetg.remove_node(SOURCE_NODE)

    return subsetg

def get_node_label(G, node):
    return str(node)

def add_node_labels(G):
    for node in G.nodes():
        G.nodes()[node]["TrueSize"] = G.nodes()[node]["cardinality"]["actual"]
        G.nodes()[node]["Subplan"] = get_node_label(G, node)

def get_query(qfn):
    qrep = load_sql_rep(qfn)
    G = qrep["subset_graph"]
    add_node_labels(G)
    return qrep

def init_datasets():
    # global dsqueries,qpaths
    # datasets = ["Simple Examples", "Join Order Benchmark", "Join Order Benchmark-M", "CEB (imdb)"]
    datasets = ["Simple Examples", "Join Order Benchmark", "Join Order Benchmark-M"]
    dsqueries = {}
    qpaths = {}

    dspaths = {}
    dspaths["Join Order Benchmark"] = os.path.join(os.path.join(QUERY_DIR, "job"), "all_job")
    dspaths["Join Order Benchmark-M"] = os.path.join(os.path.join(QUERY_DIR, "jobm"), "all_jobm")
    # dspaths["CEB (imdb)"] = os.path.join(QUERY_DIR, "ceb-imdb")
    dspaths["Simple Examples"] = os.path.join(QUERY_DIR, "debug_sqls")

    # initialize queries
    # qnames = []
    for ds in datasets:
        dsqueries[ds] = []
        dpath = dspaths[ds]
        assert os.path.exists(dpath)
        qfns = os.listdir(dpath)
        for qfn in qfns:
            if ".pkl" not in qfn:
                continue
            qpath = os.path.join(dpath, qfn)
            dsqueries[ds].append(qfn)
            qpaths[qfn] = qpath

        dsqueries[ds].sort()

    return datasets,dsqueries,qpaths
