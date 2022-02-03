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
    Button,TapTool,ActionTool,LabelSet,Label
from bokeh.events import Tap

from utils import *
# import igraph as ig

import grandalf
from grandalf.layouts import SugiyamaLayout

MIN_PLAN_NODE_SIZE=40
MAX_PLAN_NODE_SIZE=80

PLAN_COST_FMT = """
Costs using true sizes:
    Current Plan: {CPlan}
    Optimal Plan: {OptPlan}
"""

def get_pg_cost_palette():
    from bokeh.palettes import Reds256 as PAL
    PAL = list(PAL)
    PAL.reverse()
    PAL = PAL[18:-52]
    return PAL

def get_pg_edges_specs(_G, _layout):
    d = dict(xs=[], ys=[])
    for u, v, data in _G.edges(data=True):
        d['xs'].append([_layout[u][0], _layout[v][0]])
        d['ys'].append([_layout[u][1], _layout[v][1]])
    return d

def update_pg_color_bar(data):
    cms = data["figure"].select(LinearColorMapper)
    if len(cms) < 1:
        return

    assert len(cms) == 1
    costs = data["nodes_source"].data["Cost"]
    cms[0].update(low=min(costs), high=max(costs))
def get_pos_grandalf(plang):
    g = grandalf.utils.convert_nextworkx_graph_to_grandalf(plang) # undocumented function
    class defaultview(object):
        w, h = 10, 10

    for v in g.V(): v.view = defaultview()
    sug = SugiyamaLayout(g.C[0])
    sug.init_all()

    sug.draw()     # Extracts the positions
    pos = {v.data: (-v.view.xy[0], -v.view.xy[1]) for v in g.C[0].sV}
    return pos

def pg_update_query_plan(data, G, controldata):
    plang = plangraph_to_querygraph(G, controldata)

    # pos_dot = nx.nx_pydot.pydot_layout(plang, prog="dot")
    # pos = get_pos_igraph(plang)
    pos = get_pos_grandalf(plang)
    ordered_nodes, nodes_coordinates = zip(*sorted(pos.items()))
    nodes_xs, nodes_ys = list(zip(*nodes_coordinates))

    truesizes = []
    estsizes = []
    subplans = []
    labels = []

    # nodesizes = []
    xls = []
    yls = []
    costs = []

    truecost = 0.0

    # TODO: don't use G.nodes() here

    for ni, n in enumerate(ordered_nodes):
        truesizes.append(G.nodes()[n]["cardinality"]["actual"])
        estsizes.append(G.nodes()[n]["cardinality"]["curest"])
        subplans.append(G.nodes()[n]["Subplan"])

        labels.append(plang.nodes()[n]["node_label"])
        costs.append(plang.nodes()[n]["cur_cost"])
        truecost += plang.nodes()[n]["true_cost"]

        if len(n) == 1:
            xls.append(nodes_xs[ni]-0.5)
            yls.append(nodes_ys[ni]-2.0)
        else:
            xls.append(nodes_xs[ni]-0.5)
            yls.append(nodes_ys[ni]-1.0)

    nodesizes = get_node_sizes(estsizes, MIN_PLAN_NODE_SIZE,
            MAX_PLAN_NODE_SIZE)


    data["nodes_source"].data = dict(x=nodes_xs, y=nodes_ys, nodes=ordered_nodes,
                                    xl=xls,
                                    yl=yls,
                                    Cost = costs,
                                    Subplan=subplans,
                                    TrueSize=truesizes,
                                    EstimatedSize=estsizes,
                                    Label=labels,
                                    # NodeColor=node_colors,
                                    NodeSize=nodesizes,
                                    # Alphas = alphas,
                                    )
    data["edges_source"].data = get_pg_edges_specs(plang, pos)
    data["plancost"].text = PLAN_COST_FMT.format(CPlan = millify(truecost),
                                                 OptPlan = millify(plang.graph["opt_pathcost"]))

    update_pg_color_bar(data)

def init_pg_query_plan(G):
    data = {}

    p = figure(title='', height=p1_height, width=p1_width,
            tools="pan,wheel_zoom,save,reset",
            active_scroll='wheel_zoom',
            # toolbar_location=None,
            sizing_mode='scale_both')

    nodes_source = ColumnDataSource(dict(x=[], y=[], xl=[], yl=[],
                                    nodes=[],
                                    Subplan=[],
                                    Cost=[],
                                    Label=[],
                                    TrueSize=[],
                                    EstimatedSize=[],
                                    # NodeColor=[],
                                    NodeSize=[],
                                    ))
    edges_source = ColumnDataSource(dict(xs=[], ys=[]))

    edges = p.multi_line('xs', 'ys',
                        line_width=4.0,
                        color = "black",
                        alpha = 1.0,
                        source=edges_source,
                        level="underlay")

    color_mapper = LinearColorMapper(palette=get_pg_cost_palette(),
                                          low = 1,
                                          high= 1e6)

    nodes = p.circle('x', 'y', source=nodes_source,
                          size='NodeSize',
                          line_width=4.0,
                          line_color="black",
                          # color = "NodeColor",
                          # alpha="Alphas",
                          # color = "blue",
                          color={'field': "Cost", 'transform': color_mapper},
                          alpha = 1.0,
                          line_alpha = 1.0,
                          level='underlay')

    labels = LabelSet(x='xl', y='yl', text='Label', source=nodes_source,
                  background_fill_color='white', background_fill_alpha=0,
                  text_font_size="14pt", text_font_style="bold",
                  level="annotation"
                  )
    p.renderers.append(labels)


    node_hov = HoverTool(tooltips=[("Subplan", "@Subplan"), ("True Size", "@TrueSize"),
                               ("Estimated Size", "@EstimatedSize")
                               ],
                               renderers=[nodes], anchor="center",
                               attachment="above",
                               )
    p.add_tools(node_hov)

    color_bar = ColorBar(color_mapper=color_mapper, ticker= BasicTicker(),
                         location=(0,0), title="Costs", name="cost")
    p.add_layout(color_bar, 'below')

    p.axis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    plancost = Label(x=0, y=p1_height, x_units='screen', y_units='screen',
                 text=PLAN_COST_FMT,
                 render_mode='css',
                 border_line_color=None, border_line_alpha=0.0,
                 background_fill_alpha=0.0)

    p.add_layout(plancost)

    data["figure"] = p
    data["nodes_source"] = nodes_source
    data["edges_source"] = edges_source
    data["plancost"] = plancost

    return data
