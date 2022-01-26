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

from utils import *
from plangraph_utils import *
from queryplan_utils import *

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
        if _G.nodes()[u][COST_MODEL + COST_KEY + "-shortest_path"] \
            and _G.nodes()[v][COST_MODEL + COST_KEY + "-shortest_path"]:
            d["EdgeWidth"].append(12.0)
        else:
            d["EdgeWidth"].append(3.0)

    return d

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

def update():
    G = update_plangraph_properties()
    update_query_plan(qp_data, G)

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

    update()

    # G = update_plangraph_properties()
    # TODO: maybe check this condition
    # if len(p1._property_values['below']) > 1:
        # cb = p1._property_values['below'][-1]

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
    # global G
    use_postgres_ests = False
    use_true_ests = True
    for k in splan_multiples:
        splan_multiples[k] = 1.0
    # G = update_plangraph_properties()
    update()
    size_slider.value = 1.0

def update_reset_pg(new):
    global use_postgres_ests, use_true_ests
    global splan_multiples
    # global G
    use_postgres_ests = True
    use_true_ests = False
    for k in splan_multiples:
        splan_multiples[k] = 1.0
    # G = update_plangraph_properties()
    update()
    size_slider.value = 1.0

def update_cards_to_use(new):
    # global G
    # G = update_plangraph_properties()
    update()

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
    # , G
    # assert SEL_NODE_LABEL in splan_multiples
    if SEL_NODE_LABEL not in labels_to_nodes:
        return
    sel_node = labels_to_nodes[SEL_NODE_LABEL]
    assert sel_node in splan_multiples

    splan_multiples[sel_node] = size_slider.value
    # G = update_plangraph_properties()
    update()

def update_plangraph_properties():
    '''
    should handle all the updating based on changes to splan_multiples.
    --> updates nodes_source and cost_edges_source ColumnDataSources, based on which plot is made.
    '''
    # global G
    curG = G
    y = {}
    # TODO: this can be computed just once and done
    truey = {}

    ests = []
    node_colors = []
    alphas = []

    tcostkey = COST_MODEL + TRUE_COST_KEY
    need_true_costs = True
    for edge in curG.edges():
        if tcostkey in curG.edges()[edge]:
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
            y[node] = curG.nodes()[node]["cardinality"]["expected"] * mul
        else:
            y[node] = curG.nodes()[node]["cardinality"]["actual"] * mul

        ests.append(y[node])
        # updating in the graph too, so we won't need to recompute these
        curG.nodes()[node]["cardinality"]["curest"] = y[node]

        if need_true_costs:
            truey[node] = curG.nodes()[node]["cardinality"]["actual"]

    if len(truey) != 0:
        # update_costs(curG, COST_MODEL, truey, TRUE_COST_KEY)
        curG = get_shortest_path(curG, COST_MODEL, truey, TRUE_COST_KEY)

    sizes = get_node_sizes(ests, MIN_RADIUS, MAX_RADIUS)
    curG = get_shortest_path(curG, COST_MODEL, y, COST_KEY)

    for ni, node in enumerate(ordered_nodes):
        if curG.nodes()[node][COST_MODEL + COST_KEY + "-shortest_path"]:
            node_colors.append("lightgreen")
            alphas.append(1.0)
        else:
            node_colors.append("grey")
            alphas.append(1.0)

        # curG.nodes()[node]["NodeSize"] = sizes[ni]

    # this is responsible for updating the displayed plangraph graph
    nodes_source.data = dict(x=nodes_xs, y=nodes_ys, nodes=ordered_nodes,
                                    Subplan=nodedict["Subplan"],
                                    TrueSize=nodedict["TrueSize"],
                                    EstimatedSize=ests,
                                    NodeColor=node_colors,
                                    NodeSize=sizes,
                                    Alphas = alphas,
                                    )
    cost_edges_source.data = get_edges_specs(curG, layout)
    flow_edges_source.data = get_flow_specs(curG, layout)


    return curG

## state of the query being displayed
qrep = None
G = None
splan_multiples = {}
# only needs to be done once
layout = None
ordered_nodes, nodes_coordinates = None,None
nodes_xs, nodes_ys = None,None
SEL_NODE_LABEL = None
labels_to_nodes = None
nodedict = None
show_flows = False

datasets,dsqueries,qpaths = init_datasets()
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
        tools="pan,wheel_zoom,save,reset",
        active_scroll='wheel_zoom',
        # toolbar_location=None,
        sizing_mode='scale_both')

# TODO: sort by length
# nodeselector = Select(title="Click on node (or select subplan) to increase / decrease size",
        # value="", options=[], width=600)
nodeselector = Select(value="", options=[], width=600)

# TODO: reset option here

nodeselector.on_change('value', update_selected)

## every time G is updated / cards are updated, we want to find the new path
## query plan graph things
qp_data = init_cost_model_query_plan(G)

update_dataset(None, None, None)

r_circles = p1.circle('x', 'y', source=nodes_source,
                      line_color = "black",
                      line_width = 4.0,
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

color_mapper = LinearColorMapper(palette=get_plangraph_cost_palette(),
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
# p1.outline_line_color = None

size_slider = Slider(title=None, value=1, start=-10000, end=10000, step=1,
	width=p1_width)
size_slider.on_change('value', update_slider)

update_selected(None,None,None)

dummy = RadioButtonGroup(labels=[])

tab1 = Panel(child=p1, title="Plan Graph")


tab2 = Panel(child=qp_data["figure"], title="Query Plan")
tabs = Tabs(tabs=[tab1, tab2])

l = row(column(dselector, qselector, cmselector, info_tabs),
    column(
	row(dummy,
        column(checkbox_button_group, cbox_cards_to_use), spacing=500),
	tabs,
	row(nodeselector, reset_card_button, reset_pg_button),
	size_slider))

curdoc().add_root(l)

