import bokeh

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
