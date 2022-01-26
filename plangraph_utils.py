import bokeh

def get_node_sizes(ests, min_size, max_size):
    new_range = max_size-min_size
    old_range = max(ests) - min(ests)
    sizes = []
    for ni,est in enumerate(ests):
        #NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
        size = (((ests[ni] - min(ests)) * new_range) / old_range) + min_size
        sizes.append(size)

    return sizes

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
