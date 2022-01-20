from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.plotting import figure, from_networkx, curdoc
from bokeh.models import Rect, Circle, HoverTool,  TextInput, BoxZoomTool, ResetTool
from bokeh.models import LinearColorMapper, BasicTicker, ColorBar,CustomJSHover,CustomJS
from bokeh.models import Paragraph,Div
from bokeh.io import output_file
from bokeh.models import CheckboxButtonGroup, RadioButtonGroup, CheckboxGroup
from bokeh.models import Panel, Tabs
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput,\
    Button,TapTool,ActionTool

from bokeh.events import Tap

import numpy as np
import networkx as nx
from collections import defaultdict
import networkx as nx
import os
import cvxpy as cp
import sqlparse
import pickle
from networkx.readwrite import json_graph

# from utils.utils import *
# from db_utils.utils import *
# from db_utils.query_storage import *

## imported functions

NILJ_CONSTANT = 0.001
MAX_JOINS = 32

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
        if len(node1) == 1:
            nilj_cost = card2 + NILJ_CONSTANT*card1
        elif len(node2) == 1:
            nilj_cost = card1 + NILJ_CONSTANT*card2
        else:
            assert False, "one of the nodes must have a single table"

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

SOURCE_NODE = tuple("s")

MIN_RADIUS = 20.0
MAX_RADIUS = 50.0

def init_datasets():
    global dsqueries,qpaths
    # datasets = ["Simple Examples", "Join Order Benchmark", "Join Order Benchmark-M", "CEB (imdb)"]
    datasets = ["Simple Examples", "Join Order Benchmark", "Join Order Benchmark-M"]

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

    return datasets

def get_node_label(G, node):
    return str(node)

def add_node_labels(G):
    for node in G.nodes():
        G.nodes()[node]["TrueSize"] = G.nodes()[node]["cardinality"]["actual"]
        G.nodes()[node]["Subplan"] = get_node_label(G, node)

def get_flows(subsetg, cost_key):
    # TODO: add options for changing beta; look at old code
    # FIXME: assuming subsetg  is S->D; construct_lp expects it to be D->S.
    edges, costs, A, b, G, h = construct_lp(subsetg, cost_key=cost_key)
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
    prob.solve(verbose=False)
    flows = np.array(x.value)

    return flows, edges


def get_node_sizes(ests, min_size, max_size):
    new_range = max_size-min_size
    old_range = max(ests) - min(ests)
    sizes = []
    for ni,est in enumerate(ests):
        #NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
        size = (((ests[ni] - min(ests)) * new_range) / old_range) + min_size
        sizes.append(size)

    return sizes

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


def get_query(qfn):
    qrep = load_sql_rep(qfn)
    G = qrep["subset_graph"]
    add_node_labels(G)
    return qrep

def get_flow_specs(_G, _layout):
    d = dict(xs=[], ys=[], flow=[])

    if cbox_cards_to_use.active == 0:
        flow_key = COST_KEY + "flow"
    elif cbox_cards_to_use.active == 1:
        flow_key = TRUE_COST_KEY + "flow"
    else:
        assert False

    for u, v, data in _G.edges(data=True):
        d['xs'].append([_layout[u][0]+5.0, _layout[v][0]+5.0])
        d['ys'].append([_layout[u][1], _layout[v][1]])
        d["flow"].append(data[flow_key])

    return d

def get_edges_specs(_G, _layout):
    d = dict(xs=[], ys=[], join=[], updated_cost=[], EdgeWidth=[])

    if cbox_cards_to_use.active == 0:
        cost_key = COST_MODEL + COST_KEY
    elif cbox_cards_to_use.active == 1:
        cost_key = COST_MODEL + TRUE_COST_KEY

    for u, v, data in _G.edges(data=True):
        d['xs'].append([_layout[u][0], _layout[v][0]])
        d['ys'].append([_layout[u][1], _layout[v][1]])
        d["join"].append("Join")
        d["updated_cost"].append(data[cost_key])

        # selecting width based on the estimated shortest path
        if G.nodes()[u][COST_MODEL + COST_KEY + "-shortest_path"] \
            and G.nodes()[v][COST_MODEL + COST_KEY + "-shortest_path"]:
            d["EdgeWidth"].append(12.0)
        else:
            d["EdgeWidth"].append(3.0)

    return d

def get_cost_palette():
    # from bokeh.palettes import Viridis10 as PAL
    from bokeh.palettes import Reds256 as PAL
    PAL = list(PAL)
    PAL.reverse()
    PAL = PAL[18:]
    return PAL

def get_flow_palette():
    from bokeh.palettes import Blues256 as PAL
    PAL = list(PAL)
    PAL.reverse()
    PAL = PAL[18:]
    return PAL

def calc_sel_data():
    selxs = []
    selys = []
    selsizes = []

    for ni, node in enumerate(ordered_nodes):
        if G.nodes()[node]["Subplan"] != SEL_NODE_LABEL:
            continue
        selxs.append(nodes_xs[ni])
        selys.append(nodes_ys[ni])
        nodesize = nodes_source.data["NodeSize"][ni] + 6.0
        selsizes.append(nodesize)

    sel_source.data = dict(x=selxs, y=selys, nodes=["selected"],
                                NodeSize=selsizes
                                )

def update_node_highlight(event):
    nodes_clicked_ints = nodes_source.selected.indices
    if len(nodes_clicked_ints) == 0:
        return

    sel_node = ordered_nodes[nodes_clicked_ints[-1]]
    nodeselector.value = get_node_label(G, sel_node)
    update_selected(None,None,None)

def update_query(attr, old, new):
    global qrep, G, splan_multiples,layout,ordered_nodes,\
            nodes_coordinates,nodes_xs,nodes_ys,labels_to_nodes,\
            SEL_NODE_LABEL,nodedict

    curq = qselector.value

    qfn = qpaths[curq]
    qrep = get_query(qfn)

    # G will be global
    G = qrep["subset_graph"]
    if SOURCE_NODE in G.nodes():
        G.remove_node(SOURCE_NODE)

    for node in G.nodes():
        splan_multiples[node] = 1.0

    # layout = nx.nx_pydot.pydot_layout(G , prog='dot')
    layout_fn = qfn.replace("./queries", "./layouts")
    assert os.path.exists(layout_fn)
    with open(layout_fn, "rb") as f:
        layout = pickle.load(f)

    ordered_nodes, nodes_coordinates = zip(*sorted(layout.items()))
    nodes_xs, nodes_ys = list(zip(*nodes_coordinates))

    labels_to_nodes = {}
    for node in ordered_nodes:
        labels_to_nodes[get_node_label(G, node)] = node

    # TODO: ideally, we'll have something be pre-selected always
    init_selected()
    sel_source.data = dict(x=[], y=[], nodes=[], NodeSize=[])

    # SEL_NODE_LABEL = get_node_label(G, ordered_nodes[0])
    # update_selected(None,None,None)

    nodedict = defaultdict(list)
    for node in ordered_nodes:
        nodedict["Subplan"].append(G.nodes()[node]["Subplan"])
        nodedict["TrueSize"].append(G.nodes()[node]["TrueSize"])

    update_plangraph_properties()

    # TODO: maybe check this condition
    # if len(p1._property_values['below']) > 1:
        # cb = p1._property_values['below'][-1]

    # color_bar.name = "cost"
    # color_bar2.name = "flow"

    update_color_bar()

    sql = sqlparse.format(qrep["sql"], reindent=True, keyword_case='upper')
    sql = sql.replace("\n", "<br>")
    sql = sql.replace(" ", "&nbsp;")
    sqltext.text = sql

def update_color_bar():
    cms = p1.select(LinearColorMapper)
    for cm in cms:
        if cm.high > 1.0:
            low = min(cost_edges_source.data["updated_cost"])
            high= max(cost_edges_source.data["updated_cost"])
        elif cm.high <= 1.0:
            low = min(flow_edges_source.data["flow"])
            high= max(flow_edges_source.data["flow"])
        else:
            print("color bar name: ", cm.name)
            print(cm.low, cm.high)
            continue

        cm.update(low=low, high=high)

def update_dataset(attr, old, new):
    ds = dselector.value
    print("Update dataset ", ds)

    qselector.options = dsqueries[ds]
    qselector.value = qselector.options[0]

    update_query(None, None, None)

def init_selected():
    selectoroptions = []
    for node in ordered_nodes:
        selectoroptions.append(get_node_label(G, node))

    nodeselector.options = selectoroptions

def update_selected(attr, old, new):
    global SEL_NODE_LABEL
    SEL_NODE_LABEL = nodeselector.value
    if SEL_NODE_LABEL == "":
        return

    calc_sel_data()

    # TODO: update the slider based on it too
    sel_node = labels_to_nodes[SEL_NODE_LABEL]
    assert sel_node in splan_multiples
    size_slider.value = splan_multiples[sel_node]

def update_reset_true(new):
    global use_postgres_ests, use_true_ests
    global splan_multiples
    use_postgres_ests = False
    use_true_ests = True
    for k in splan_multiples:
        splan_multiples[k] = 1.0
    update_plangraph_properties()
    size_slider.value = 1.0

def update_reset_pg(new):
    global use_postgres_ests, use_true_ests
    global splan_multiples
    use_postgres_ests = True
    use_true_ests = False
    for k in splan_multiples:
        splan_multiples[k] = 1.0
    update_plangraph_properties()
    size_slider.value = 1.0

def update_cards_to_use(new):
    update_plangraph_properties()

def update_edges(new):
    # get the lines
    plain_edges_lines.visible = False
    flow_edges_lines.visible = False
    cost_edges_lines.visible = False
    color_bar.visible = False
    color_bar2.visible = False

    for sel in new:
        if sel == 0:
            cost_edges_lines.visible = True
            color_bar.visible = True
        if sel == 1:
            flow_edges_lines.visible = True
            color_bar2.visible = True

    if len(new) == 0:
        plain_edges_lines.visible = True


def update_slider(attr, old, new):
    global splan_multiples
    # assert SEL_NODE_LABEL in splan_multiples
    if SEL_NODE_LABEL not in labels_to_nodes:
        return
    sel_node = labels_to_nodes[SEL_NODE_LABEL]
    assert sel_node in splan_multiples

    splan_multiples[sel_node] = size_slider.value
    update_plangraph_properties()

def update_plangraph_properties():
    '''
    should handle all the updating based on changes to splan_multiples.
    --> updates nodes_source and cost_edges_source ColumnDataSources, based on which plot is made.
    '''
    global G, nodes_source, cost_edges_source,splan_multiples

    y = {}
    # TODO: this can be computed just once and done
    truey = {}

    ests = []
    node_colors = []
    alphas = []

    tcostkey = COST_MODEL + TRUE_COST_KEY
    need_true_costs = True
    for edge in G.edges():
        if tcostkey in G.edges()[edge]:
            need_true_costs = False
            break

    for node in ordered_nodes:
        s = splan_multiples[node]
        if s < -0.001:
            mul = 1 / float(abs(s))
        elif s > 0.001:
            mul = s
        else:
            mul = 1.0

        if use_postgres_ests:
            y[node] = G.nodes()[node]["cardinality"]["expected"] * mul
        else:
            y[node] = G.nodes()[node]["cardinality"]["actual"] * mul

        ests.append(y[node])

        if need_true_costs:
            truey[node] = G.nodes()[node]["cardinality"]["actual"]

    if len(truey) != 0:
        # update_costs(G, COST_MODEL, truey, TRUE_COST_KEY)
        G = get_shortest_path(G, COST_MODEL, truey, TRUE_COST_KEY)

    sizes = get_node_sizes(ests, MIN_RADIUS, MAX_RADIUS)
    G = get_shortest_path(G, COST_MODEL, y, COST_KEY)

    for ni, node in enumerate(ordered_nodes):
        if G.nodes()[node][COST_MODEL + COST_KEY + "-shortest_path"]:
            node_colors.append("black")
            alphas.append(1.0)
        else:
            node_colors.append("grey")
            alphas.append(1.0)

        G.nodes()[node]["NodeSize"] = sizes[ni]

    nodes_source.data = dict(x=nodes_xs, y=nodes_ys, nodes=ordered_nodes,
                                    Subplan=nodedict["Subplan"],
                                    TrueSize=nodedict["TrueSize"],
                                    EstimatedSize=ests,
                                    NodeColor=node_colors,
                                    NodeSize=sizes,
                                    Alphas = alphas,
                                    )
    cost_edges_source.data = get_edges_specs(G, layout)
    flow_edges_source.data = get_flow_specs(G, layout)

info_width = 300

p1_width = 1000
p1_height = 600

## state of the query being displayed
qrep = None
G = None
splan_multiples = {}
qpaths = {}
dsqueries = {}
# only needs to be done once
layout = None
ordered_nodes, nodes_coordinates = None,None
nodes_xs, nodes_ys = None,None
SEL_NODE_LABEL = None
labels_to_nodes = None
nodedict = None
show_flows = False

datasets = init_datasets()

dselector = Select(title="Dataset", value="Simple Examples", options=datasets,
        width=info_width)
qselector = Select(title="Query", value="", options=[], width=info_width)
cmselector = Select(title="Cost Model", value="", options=["C"],
        width=info_width)

dselector.on_change("value", update_dataset)
qselector.on_change("value", update_query)

edge_labels = ["Edge Costs", "Edge Flows"]
checkbox_button_group = CheckboxGroup(labels=edge_labels,
        active=[0],
        width=200)
checkbox_button_group.on_click(update_edges)

cbox2_labels = ["Use estimated sizes", "Use true sizes"]
cbox_cards_to_use = RadioButtonGroup(labels=cbox2_labels,
        active=0,
        width=200)
cbox_cards_to_use.on_click(update_cards_to_use)

use_postgres_ests = False
use_true_ests = True
reset_card_button = Button(label="Reset all to true sizes",
    button_type="success")
reset_pg_button = Button(label="Reset all to PostgreSQL estimates",
    button_type="success")

reset_card_button.on_click(update_reset_true)
reset_pg_button.on_click(update_reset_pg)

nodes_source = ColumnDataSource(dict(x=[], y=[],
                                nodes=[],
                                Subplan=[],
                                TrueSize=[],
                                EstimatedSize=[],
                                NodeColor=[], NodeSize=[], Alphas=[],
                                ))
sel_source = ColumnDataSource(dict(x=[], y=[], NodeSize=[]))
cost_edges_source = ColumnDataSource(dict(xs=[], ys=[], join=[],
                        updated_cost=[], EdgeWidth=[]))
flow_edges_source = ColumnDataSource(dict(xs=[], ys=[],
                        flow=[]))

## info text
p2 = figure(title='',
        height=600,
        width=info_width,
        toolbar_location=None)

p2.axis.visible = False
p2.xgrid.grid_line_color = None
p2.ygrid.grid_line_color = None
p2.outline_line_color = None

# sqltext = Paragraph(text = "", width = 300,
        # height_policy = "auto", style =
        # {'fontsize': '10pt', 'color': 'black', 'font-family': 'arial'})

sqltext = Div(text="",
        style={'font-size': '12pt', 'color': 'black'},
        width=info_width)

tab_info1 = Panel(child=sqltext, title="SQL")
tab_info2 = Panel(child=p2, title="Join Graph")
info_tabs = Tabs(tabs=[tab_info1, tab_info2])

## actual bokeh plotting
p1 = figure(title='', height=p1_height, width=p1_width,
        toolbar_location=None,
        sizing_mode='scale_both')

# TODO: sort by length
# nodeselector = Select(title="Click on node (or select subplan) to increase / decrease size",
        # value="", options=[], width=600)
nodeselector = Select(value="", options=[], width=600)

# TODO: reset option here

nodeselector.on_change('value', update_selected)

update_dataset(None, None, None)

r_circles = p1.circle('x', 'y', source=nodes_source,
                      size='NodeSize',
                      color = "NodeColor",
                      alpha="Alphas",
                      level='overlay')

p1.add_tools(TapTool(renderers=[r_circles]))
p1.on_event(Tap, update_node_highlight)

sel_circle = p1.circle('x', 'y', source=sel_source,
                      size='NodeSize',
                      fill_color='white',
                      line_dash="dashed",
                      line_width=6.0,
                      alpha=1.0,
                      level="underlay")

color_mapper = LinearColorMapper(palette=get_cost_palette(),
                                      low = min(cost_edges_source.data["updated_cost"]),
                                      high= max(cost_edges_source.data["updated_cost"]))

cost_edges_lines = p1.multi_line('xs', 'ys',
                          line_width='EdgeWidth',
                          color={'field': "updated_cost", 'transform': color_mapper},
                          source=cost_edges_source)
color_mapper2 = LinearColorMapper(palette=get_flow_palette(),
                                      low = min(flow_edges_source.data["flow"]),
                                      high= max(flow_edges_source.data["flow"]))

plain_edges_lines = p1.multi_line('xs', 'ys', line_width=3.0,
                          color="black",
                          source=cost_edges_source)
plain_edges_lines.visible = False

flow_edges_lines = p1.multi_line('xs', 'ys', line_width=3.0,
                          color={'field': "flow", 'transform': color_mapper2},
                          source=flow_edges_source)
flow_edges_lines.visible = False

color_bar = ColorBar(color_mapper=color_mapper, ticker= BasicTicker(),
                     location=(0,0), title="Costs", name="cost")

color_bar2 = ColorBar(color_mapper=color_mapper2, ticker= BasicTicker(),
                     location=(0,0), title = "Flows", name="flow")
color_bar2.visible = False

p1.add_layout(color_bar, 'below')
p1.add_layout(color_bar2, 'left')

node_hov = HoverTool(tooltips=[("Subplan", "@Subplan"), ("True Size", "@TrueSize"),
                           ("Estimated Size", "@EstimatedSize")
                           ],
                           renderers=[r_circles], anchor="center",
                           attachment="above",
                           )
edge_hov = HoverTool(tooltips=[("Join", "@join")],
                           renderers=[cost_edges_lines], anchor="center",
                           attachment = "right",
                           )
p1.add_tools(node_hov)
p1.add_tools(edge_hov)

p1.axis.visible = False
p1.xgrid.grid_line_color = None
p1.ygrid.grid_line_color = None
p1.outline_line_color = None

size_slider = Slider(title=None, value=1, start=-10000, end=10000, step=1,
	width=p1_width)
size_slider.on_change('value', update_slider)

# div = Div(text= '<b>text0</b>', style={'font-size': '150%', 'color': 'blue'})
# str_list = ['text0', 'text1', 'text2']
# # str_slider = Slider(start=0, end=len(str_list)-1, value=0, step=1, title="string")
# callback = CustomJS(args=dict(div=div, str_list = str_list, str_slider=size_slider),
# code="""
    # const v = str_slider.value
    # div.text = str_list[0]
# """)
# size_slider.js_on_change('value', callback)

update_selected(None,None,None)

dummy = RadioButtonGroup(labels=[])

tab1 = Panel(child=p1, title="Plan Graph")

p3 = figure(width=300, height=300)
p3.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=3, color="navy",
	alpha=0.5)

# TODO: add postgresql plan as well;
tab2 = Panel(child=p3, title="Query Plan")
tabs = Tabs(tabs=[tab1, tab2])

# reset_card_button = Button("Reset all to true sizes")
# reset_pg_button = Button("Reset all to PostgreSQL estimates")

l = row(column(dselector, qselector, cmselector, info_tabs),
    column(
	row(dummy,
        column(checkbox_button_group, cbox_cards_to_use), spacing=500),
	tabs,
	row(nodeselector, reset_card_button, reset_pg_button),
	size_slider))

curdoc().add_root(l)

# output_file("plangraph.html")

#show(l, notebook_handle=True)
# output_file("plangraph.html")
# show(l)

