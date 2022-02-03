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
    Button,TapTool,ActionTool,LabelSet
from bokeh.events import Tap

from utils import *
# import igraph as ig

import grandalf
from grandalf.layouts import SugiyamaLayout

# MIN_PLAN_NODE_SIZE=50
# MAX_PLAN_NODE_SIZE=80

KEY_COLOR = "#66ccff"
JoinGraphNodeToolTip = """
    <div>
        <div>
            <span style="font-size: 10px; color: {KEYCOLOR}">
            &nbsp; Table: </span>
            <span style="font-size: 10px;">
            @Table <br> </span>

            <span style="font-size: 10px; color: {KEYCOLOR}">
            Filter: </span>
            <span style="font-size: 10px;">
            @Filter <br> </span>
        </div>
    </div>
""".format(KEYCOLOR=KEY_COLOR)

def get_qp_cost_palette():
    from bokeh.palettes import Reds256 as PAL
    PAL = list(PAL)
    PAL.reverse()
    PAL = PAL[18:-52]
    return PAL

def get_join_edges_specs(_G, _layout):
    d = dict(xs=[], ys=[])
    for u, v, data in _G.edges(data=True):
        d['xs'].append([_layout[u][0], _layout[v][0]])
        d['ys'].append([_layout[u][1], _layout[v][1]])
    return d

def update_joingraph(data, joingraph, controldata):
    pos = nx.spring_layout(joingraph, scale=0.60)
    ordered_nodes, nodes_coordinates = zip(*sorted(pos.items()))
    nodes_xs, nodes_ys = list(zip(*nodes_coordinates))

    labels = []
    nodesizes = []
    xls = []
    yls = []

    # TODO: don't use G.nodes() here
    labels = []
    filters = []
    tables = []
    aliases = []

    for ni, n in enumerate(ordered_nodes):
        nodesizes.append(50.0)
        tfmt = "{TAB} as {ALIAS}"
        labels.append(n)
        tables.append(tfmt.format(TAB=joingraph.nodes()[n]["real_name"],
                                  ALIAS=n))
        # aliases.append(n)
        pred_str = ""
        if len(joingraph.nodes()[n]["predicates"]) > 0:
            pred_str = " AND ".join(joingraph.nodes()[n]["predicates"])
        filters.append(pred_str)

        # xls.append(nodes_xs[ni])
        # yls.append(nodes_ys[ni])
        xls.append(nodes_xs[ni]-0.05)
        yls.append(nodes_ys[ni]-0.02)

    # nodesizes = get_node_sizes(estsizes, MIN_PLAN_NODE_SIZE,
            # MAX_PLAN_NODE_SIZE)

    data["edges_source"].data = get_join_edges_specs(joingraph, pos)
    data["nodes_source"].data = dict(x=nodes_xs, y=nodes_ys, nodes=ordered_nodes,
                                    xl=xls,
                                    yl=yls,
                                    Label=labels,
                                    Table=tables,
                                    Filter=filters,
                                    # Alias=aliases,
                                    # NodeColor=node_colors,
                                    NodeSize=nodesizes,
                                    # Alphas = alphas,
                                    )

def init_joingraph():
    data = {}
    HEIGHT=600
    p = figure(title='',
            height=HEIGHT,
            width=info_width,
            x_range=(-1,1),
            y_range=(-1,1),
            # tools="pan,wheel_zoom,save,reset",
            tools="wheel_zoom,save,reset",
            active_scroll='wheel_zoom',
            # toolbar_location=None,
            sizing_mode='scale_both')

    edges_source = ColumnDataSource(dict(xs=[], ys=[]))
    nodes_source = ColumnDataSource(dict(x=[], y=[], xl=[], yl=[],
                                    nodes=[],
                                    Label=[],
                                    Filter=[],
                                    # TrueSize=[],
                                    # EstimatedSize=[],
                                    # NodeColor=[],
                                    NodeSize=[],
                                    ))

    edges = p.multi_line('xs', 'ys',
                        line_width=4.0,
                        color = "black",
                        alpha = 1.0,
                        source=edges_source,
                        level="underlay")

    nodes = p.circle('x', 'y', source=nodes_source,
                          size='NodeSize',
                          line_width=4.0,
                          line_color="black",
                          color = "white",
                          alpha = 1.0,
                          line_alpha = 1.0,
                          level='underlay')

    labels = LabelSet(x='xl', y='yl', text='Label', source=nodes_source,
                  background_fill_color='white', background_fill_alpha=0,
                  text_font_size="14pt", text_font_style="bold",
                  level="annotation"
                  )
    p.renderers.append(labels)

    # node_hov = HoverTool(tooltips=[("Table", "@Table"),
                               # ],
                               # renderers=[nodes], anchor="center",
                               # attachment="above",
                               # )
    node_hov = HoverTool(tooltips=JoinGraphNodeToolTip,
                               renderers=[nodes], anchor="center",
                               attachment="above")
    p.add_tools(node_hov)

    # color_bar = ColorBar(color_mapper=color_mapper, ticker= BasicTicker(),
                         # location=(0,0), title="Costs", name="cost")
    # p.add_layout(color_bar, 'below')

    p.axis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    # p1.outline_line_color = None

    data["figure"] = p
    data["nodes_source"] = nodes_source
    data["edges_source"] = edges_source
    return data
