# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 20:57:40 2023

@author: lukas
"""
#%%
# A package for automatically drawing diagrams of PyPSA components on a bus and
# in a network. 

# The schemdraw package is required for drawing the diagrams.
# https://schemdraw.readthedocs.io/en/latest/


#%% 
def draw_bus(n, bus, show = True, 
             bus_color = 'steelblue', link_color = 'darkorange',
             fontsize = 7, title_fontsize = 12,
             line_length = 1.5):
    # Draw a bus, as well as the components on the bus.
    import schemdraw
    import schemdraw.elements as elm
    from schemdraw.segments import Segment, util, math, SegmentCircle
    
    # ----- Define custom elements -----
    class MyGen(elm.Element):
        def __init__(self, *d, **kwargs):
            super().__init__(*d, **kwargs)
            self.segments.append(Segment(
                [(0, 0), (0.75, 0)]))
            sin_y = util.linspace(-.25, .25, num=25)
            sin_x = [.2 * math.sin((sy-.25)*math.pi*2/.5) + 1.25 for sy in sin_y]
            self.segments.append(Segment(list(zip(sin_x, sin_y))))
            self.segments.append(SegmentCircle((1.25, 0), 0.5,))
            
    class MyLoad(elm.Element):
        def __init__(self, *d, **kwargs):
            super().__init__(*d, **kwargs)
            lead = 0.95
            h = 0.8
            w = 0.5
            self.segments.append(Segment(
                [(0, 0), (0, lead), (-w, lead+h), (w, lead+h), (0, lead)]))
            self.params['drop'] = (0, 0)
            self.params['theta'] = 0
            self.anchors['start'] = (0, 0)
            self.anchors['center'] = (0, 0)
            self.anchors['end'] = (0, 0)
            
    class MyStore(elm.Element):
        def __init__(self, *d, **kwargs):
            super().__init__(*d, **kwargs)
            lead = 0.75
            h = lead + 1
            w = 1
            self.segments.append(Segment(
                [(0, 0), (lead, 0), (lead, w/2), (h, w/2),
                  (h, -w/2), (lead, -w/2), (lead, 0)
                  ]))
    
    # ----- Get elements on this bus from network -----
    gens   = n.generators[n.generators['bus'] == bus] #Get all generators on bus
    loads  = n.loads[n.loads['bus'] == bus]
    stores = n.stores[n.stores['bus'] == bus]
    
    # ----- Draw bus using schemdraw -----
    with schemdraw.Drawing(show = show) as d:
        # Add initial dot
        d += elm.Dot().color(bus_color).label(bus, fontsize = title_fontsize) #Start bus
        
        
        for gen in gens.index:
            # Create a new line piece, and add icon with text
            d += elm.Line().color(bus_color).length(line_length) #Add line piece
            d.push() # Save this place
            label = gen.replace(' ', ' \n') + '\n \n p: ' + str(round(n.generators.loc[gen].p_nom_opt, 2))
            d += MyGen().up().label(label, loc='right', fontsize = fontsize)
            d.pop() # Return to saved place
        
        for store in stores.index:
            # Create a new line piece, and add icon with text
            d += elm.Line().color(bus_color).length(line_length) #Add line piece
            d.push()
            label = store.replace(' ', ' \n') + '\n \n e: ' + str(round(n.stores.loc[store].e_nom_opt, 2))
            d += MyStore().up().label(label, loc = 'right', fontsize = fontsize)
            d.pop()
            
        for load in loads.index:
            # Create a new line piece, and add icon with text
            d += elm.Line().color(bus_color).length(line_length) #Add line piece
            d.push()
            label = load.replace(' ', ' \n') + '\n \n mean p: ' + str(round(n.loads_t.p[load].mean(), 2))
            d += MyLoad().right().label(label, loc='top', fontsize = fontsize)
            d.pop()
            
        # End bus with a line ending in a dot
        d += elm.Line(arrow = '-o').color(bus_color).length(line_length) # End bus
        
    return d