import bokeh

KEY_COLOR = "#66ccff"
PlanGraphNodeToolTip = """
    <div>
        <div>
            <span style="font-size: 10px; color: {KEYCOLOR}">
            &nbsp; Subplan: </span>
            <span style="font-size: 12px; font-weight: bold">
            @Subplan <br> </span>

            <span style="font-size: 10px; color: {KEYCOLOR}">
            True Size: </span>
            <span style="font-size: 10px;">
            @TrueSize <br> </span>

            <span style="font-size: 10px; color: {KEYCOLOR}">
            &nbsp; Est Size: </span>
            <span style="font-size: 10px;">
            @EstimatedSize <br> </span>
        </div>
    </div>
""".format(KEYCOLOR=KEY_COLOR)

PlanGraphEdgeToolTip = """
    <div>
        <div>
            <span style="font-size: 10px; color: {KEYCOLOR}">
            Join  : </span>
            <span style="font-size: 10px; font-weight: bold">
            @Join <br> </span>

            <span style="font-size: 10px; color: {KEYCOLOR}">
            True Cost: </span>
            <span style="font-size: 10px;">
            @TrueCost <br> </span>

            <span style="font-size: 10px; color: {KEYCOLOR}">
            Est Cost   : </span>
            <span style="font-size: 10px;">
            @EstCost <br> </span>

            <span style="font-size: 10px; color: {KEYCOLOR}">
            True Flow: </span>
            <span style="font-size: 10px;">
            @TrueFlow <br> </span>

            <span style="font-size: 10px; color: {KEYCOLOR}">
            Est Flow: </span>
            <span style="font-size: 10px;">
            @EstFlow <br> </span>
        </div>
    </div>
""".format(KEYCOLOR=KEY_COLOR)

def get_plangraph_cost_palette():
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
