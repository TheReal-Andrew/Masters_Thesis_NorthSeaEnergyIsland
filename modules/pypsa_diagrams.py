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
def circle_points(r, n):
    # From https://stackoverflow.com/questions/33510979/generator-of-evenly-spaced-points-in-a-circle-in-python
    import numpy as np
    import pandas as pd
    
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
        
    return circles

def plot_circle_points(r, n):
    import matplotlib.pyplot as plt
    import numpy as np
    
    pos = circle_points(r, n)
    
    plt.plot(pos[0][:,0], pos[0][:,1], 'o')

#%% Drawing functions

def draw_bus(n, bus, show = True, 
             bus_color = 'steelblue', link_color = 'darkorange',
             fontsize = 7, title_fontsize = 12,
             line_length = 1.5, link_line_length = 0.75,
             handle_bi = False,
             filename = 'bus_diagram.pdf'):
    # Draw a bus, as well as the components on the bus.
    import schemdraw
    import schemdraw.elements as elm
    from schemdraw.segments import Segment, util, math, SegmentCircle
    
    n = n.copy()
    
    # ----- Handle bidirectional links -----
    if handle_bi:
        n.buses = n.buses[~n.buses.index.str.contains('e0|e1')] #Clean out buses
        n.links = n.links[~n.links.index.str.contains('e0|e1')] #Clean out links
        
        n.links['bus0'] = ['Energy Island' for x in n.links['bus0']]
        n.links['bus1'] = [x[10:-3] for x in n.links['bus1']]
    
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
    links  = n.links[(n.links.bus0 == bus) | (n.links.bus1 == bus)]
    gens   = n.generators[n.generators['bus'] == bus] #Get all generators on bus
    loads  = n.loads[n.loads['bus'] == bus]
    stores = n.stores[n.stores['bus'] == bus]
    
    # ----- Draw bus using schemdraw -----
    with schemdraw.Drawing(file = filename) as d:
        # Add initial dot
        d += elm.Dot().color(bus_color).label(bus, fontsize = title_fontsize) #Start bus
        
        for link in links.index:
            label = link.replace(' ', ' \n')
            d += elm.Line().color(bus_color).length(link_line_length) 
            d += elm.Dot().color(link_color).label(label, loc = 'bottom', fontsize = fontsize)
        
        for gen in gens.index:
            # Create a new line piece, and add icon with text
            d += elm.Line().color(bus_color).length(line_length) #Add line piece
            d.push() # Save this place
            label = gen.replace(' ', ' \n')
            # label = gen.replace(' ', ' \n') + '\n \n p: ' + str(round(n.generators.loc[gen].p_nom_opt, 2))
            d += MyGen().up().label(label, loc='right', fontsize = fontsize)
            d.pop() # Return to saved place
        
        for store in stores.index:
            # Create a new line piece, and add icon with text
            d += elm.Line().color(bus_color).length(line_length) #Add line piece
            d.push()
            label = 'Store'
            # label = store.replace(' ', ' \n') + '\n \n e: ' + str(round(n.stores.loc[store].e_nom_opt, 2))
            d += MyStore().up().label(label, loc = 'right', fontsize = fontsize)
            d.pop()
            
        for load in loads.index:
            # Create a new line piece, and add icon with text
            d += elm.Line().color(bus_color).length(line_length) #Add line piece
            d.push()
            label = load.replace(' ', ' \n')
            # label = load.replace(' ', ' \n') + '\n \n mean p: ' + str(round(n.loads_t.p[load].mean(), 2))
            d += MyLoad().right().label(label, loc='top', fontsize = fontsize)
            d.pop()
        
        # End bus with a line ending in a dot
        d += elm.Line(arrow = '-o').color(bus_color).length(line_length) # End bus
        
        
        
    return d

#%% DRAW NETWORK --------------------------------------------------------------
def draw_network(n, spacing = 2, 
                 line_length = 1.5, link_line_length = 0.75, 
                 headwidth = 0.45, headlength = 0.75,
                 fontsize = 8, title_fontsize = 12,
                 bus_color = 'steelblue', 
                 component_color = 'black',
                 link_color = 'darkorange', 
                 arrow_color = 'darkorange',
                 theme = 'default',
                 pos = None, filename = 'graphics/pypsa_diagram.pdf',
                 handle_bi = False, index1 = None,
                 show_country_values = False, exclude_bus = '',
                 ):
    import pandas as pd
    pd.options.mode.chained_assignment = None #Disable warning (For line 218)
    import numpy as np
    import schemdraw
    import schemdraw.elements as elm
    import matplotlib.pyplot as plt
    
    schemdraw.theme(theme)
    
    plt.figure()
    n = n.copy()
    
    if not index1 == None:
        n.links = n.links.reindex(index1)
    
    
    # ----- Handle bidirectional links -----
    if handle_bi:
        n.links = n.links.loc[n.links.bus0 == "Energy Island"]
        
    
    # ----- Set positions -----
    if pos == None or len(pos) != len(n.buses.index):
        #Determine if automatic position is used, or if positions are provided
        print('\nWARNING: draw_network():  No position given, or not sufficienct positions for the buses. Using automatic circular layout. \n')
        
        n_buses = len(n.buses.index)
        T = [n_buses]
        R = [n_buses * spacing]
    
        # Create positions in a circle
        pos = circle_points(R, T)
        s = pd.DataFrame(pos[0], columns = ['x', 'y'])
        s.index = n.buses.index
        
    else:
        print('\nINFO: draw_network(): bus positions given. spacing parameter has no effects. \n')
        s = pd.DataFrame(pos, columns = ['x', 'y'])
        s.index = n.buses.index

    # pdiag.plot_circle_points(R, T)

    # ------ CUSTOM ELEMENTS --------------------------------------------------
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
            
    # ------ DRAW NETWORK -----------------------------------------------------
    with schemdraw.Drawing(file = filename) as d:
        
        # Add columns with start and end cooridnates for links
        n.links['start'] = np.nan
        n.links['end']   = np.nan
        
        for bus in n.buses.index:
            d += (elm.Dot()
                  .color(bus_color)
                  .label(bus, fontsize = title_fontsize)
                  .at(( s['x'][bus], s['y'][bus] )))
            
            # ----- Get elements on this bus from network -----
            gens   = n.generators[n.generators['bus'] == bus] #Get all generators on bus
            loads  = n.loads[n.loads['bus'] == bus]
            stores = n.stores[n.stores['bus'] == bus]
            
            for link in n.links.index:
                # Loop through 
                
                if n.links['bus0'][link] == bus:
                    
                    d += elm.Line().color(bus_color).length(link_line_length) #Add line piece
                    d += (C := elm.Dot().color(link_color))
                    
                    n.links['start'][link] = C
            
            for gen in gens.index:
                # Loop through generators on this bus, and add new line segment
                # and generator icon. Add label to icon.
                
                d += elm.Line().color(bus_color).length(line_length) #Add line piece
                d.push() # Save position
                
                value_string = f'\n \n p: {round(n.generators.loc[gen].p_nom_opt, 2)}'
                label_addition = value_string if show_country_values else ''
                label_addition = value_string if bus == exclude_bus else label_addition
                
                label = (gen.replace(' ', ' \n') 
                          + label_addition
                         )
                d += MyGen().up().label(label, loc='right', fontsize = fontsize)
                d.pop()  # Return to saved position
            
            for store in stores.index:
                # Loop through stores on this bus, and add new line segment
                # and store icon. Add label to icon.
                
                d += elm.Line().color(bus_color).length(line_length) #Add line piece
                d.push()
                
                value_string = f'\n \n e: {round(n.stores.loc[store].e_nom_opt, 2)}'
                label_addition = value_string if show_country_values and bus != exclude_bus else ''
                label_addition = value_string if bus == exclude_bus else label_addition
                
                label = (store.replace(' ', ' \n') 
                          + label_addition
                         )
                d += MyStore().up().label(label, loc = 'right', fontsize = fontsize).color(component_color)
                d.pop()
                
            for load in loads.index:
                # Loop through loads on this bus, and add new line segment
                # and load icon. Add label to icon.
                
                d += elm.Line().color(bus_color).length(line_length) #Add line piece
                d.push()
                
                value_string = f'\n \n p: {round(n.loads_t.p[load].mean(), 2)}'
                label_addition = value_string if show_country_values and bus != exclude_bus else ''
                label_addition = value_string if bus == exclude_bus else label_addition
                
                label = (load.replace(' ', ' \n') 
                          + label_addition
                         )
                d += MyLoad().right().label(label, loc='top', fontsize = fontsize).color(component_color)
                d.pop()
                
            for link in n.links.index:
                
                if n.links['bus1'][link] == bus:
                    
                    d += elm.Line().color(bus_color).length(link_line_length) #Add line piece
                    d += (C := elm.Dot().color(link_color))
                    
                    n.links['end'][link] = C
            
            d += elm.Line(arrow = '-o').color(bus_color).length(link_line_length)
        
        
        w = ( (n.links.p_nom_opt/n.links.p_nom_opt.max()) )*4 + 0.05
        
        for link in n.links.index:
            # Loop through all links, and create lines with arrows.
            
            n.links.reindex(index1)
            
            w_link = w[link]
            
            style = 'N'
            
            d += ( elm.Wire(style, k = 1)
                  .color(link_color)
                  .at(n.links['start'][link].center)
                  .to(n.links['end'][link].center)
                   .label('p: ' + str(round(n.links.p_nom_opt[link],2)), 
                          fontsize = title_fontsize,
                          color = bus_color)
                  .zorder(0.1)
                  .linewidth(w_link)
                  )
            
            d += elm.Arrowhead(headwidth = headwidth, headlength = headlength).color(arrow_color)
            
            if n.links.p_min_pu[link] < 0:
                # if link is bidirectional, add an additional arrow head.
                d += ( elm.Wire(style, k = 1)
                      .color(link_color)
                      .at(n.links['end'][link].center)
                      .to(n.links['start'][link].center)
                      .zorder(0)
                      .linewidth(w_link)
                      )
                
                d += elm.Arrowhead(headwidth = headwidth, headlength = headlength).color(arrow_color)
    
    fig = d.draw()
    
    return fig

#%% DRAW NETWORK --------------------------------------------------------------

def draw_network_old(n, spacing = 2, 
                 line_length = 1.5, link_line_length = 0.75, 
                 headwidth = 0.45, headlength = 0.75,
                 fontsize = 8, title_fontsize = 12,
                 bus_color = 'steelblue', 
                 component_color = 'black',
                 link_color = 'darkorange', 
                 arrow_color = 'darkorange',
                 theme = 'default',
                 pos = None, filename = 'graphics/pypsa_diagram.pdf',
                 handle_bi = False,
                 index1 = None,
                 ):
    import pandas as pd
    pd.options.mode.chained_assignment = None #Disable warning (For line 218)
    import numpy as np
    import schemdraw
    import schemdraw.elements as elm
    import matplotlib.pyplot as plt
    
    schemdraw.theme(theme)
    
    plt.figure()
    n = n.copy()
    
    if not index1 == None:
        
        n.links = n.links.reindex(index1)
    
    
    # ----- Handle bidirectional links -----
    if handle_bi:
        n.buses = n.buses[~n.buses.index.str.contains('e0|e1')] #Clean out buses
        n.links = n.links[~n.links.index.str.contains('e0|e1')] #Clean out links
        
        n.links['bus0'] = ['Energy Island' for x in n.links['bus0']]
        n.links['bus1'] = [x[10:-3] for x in n.links['bus1']]
        
    
    # ----- Set positions -----
    if pos == None or len(pos) != len(n.buses.index):
        #Determine if automatic position is used, or if positions are provided
        print('\nWARNING: draw_network():  No position given, or not sufficienct positions for the buses. Using automatic circular layout. \n')
        
        n_buses = len(n.buses.index)
        T = [n_buses]
        R = [n_buses * spacing]
    
        # Create positions in a circle
        pos = circle_points(R, T)
        s = pd.DataFrame(pos[0], columns = ['x', 'y'])
        s.index = n.buses.index
        
    else:
        print('\nINFO: draw_network(): bus positions given. spacing parameter has no effects. \n')
        s = pd.DataFrame(pos, columns = ['x', 'y'])
        s.index = n.buses.index

    # pdiag.plot_circle_points(R, T)

    # ------ CUSTOM ELEMENTS --------------------------------------------------
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
            
    # ------ DRAW NETWORK -----------------------------------------------------
    with schemdraw.Drawing(file = filename) as d:
        
        # Add columns with start and end cooridnates for links
        n.links['start'] = np.nan
        n.links['end']   = np.nan
        
        for bus in n.buses.index:
            d += (elm.Dot()
                  .color(bus_color)
                  .label(bus, fontsize = title_fontsize)
                  .at(( s['x'][bus], s['y'][bus] )))
            
            # ----- Get elements on this bus from network -----
            gens   = n.generators[n.generators['bus'] == bus] #Get all generators on bus
            loads  = n.loads[n.loads['bus'] == bus]
            stores = n.stores[n.stores['bus'] == bus]
            
            for link in n.links.index:
                # Loop through 
                
                if n.links['bus0'][link] == bus:
                    
                    d += elm.Line().color(bus_color).length(link_line_length) #Add line piece
                    d += (C := elm.Dot().color(link_color))
                    
                    n.links['start'][link] = C
            
            for gen in gens.index:
                # Loop through generators on this bus, and add new line segment
                # and generator icon. Add label to icon.
                
                d += elm.Line().color(bus_color).length(line_length) #Add line piece
                d.push() # Save position
                label = (gen.replace(' ', ' \n') 
                          + '\n \n p: ' + str(round(n.generators.loc[gen].p_nom_opt, 2))
                         )
                d += MyGen().up().label(label, loc='right', fontsize = fontsize)
                d.pop()  # Return to saved position
            
            for store in stores.index:
                # Loop through stores on this bus, and add new line segment
                # and store icon. Add label to icon.
                
                d += elm.Line().color(bus_color).length(line_length) #Add line piece
                d.push()
                label = (store.replace(' ', ' \n') 
                          + '\n \n e: ' + str(round(n.stores.loc[store].e_nom_opt, 2))
                         )
                d += MyStore().up().label(label, loc = 'right', fontsize = fontsize).color(component_color)
                d.pop()
                
            for load in loads.index:
                # Loop through loads on this bus, and add new line segment
                # and load icon. Add label to icon.
                
                d += elm.Line().color(bus_color).length(line_length) #Add line piece
                d.push()
                label = (load.replace(' ', ' \n') 
                          + '\n \n mean p: ' + str(round(n.loads_t.p[load].mean(), 2))
                         )
                d += MyLoad().right().label(label, loc='top', fontsize = fontsize).color(component_color)
                d.pop()
                
            for link in n.links.index:
                
                if n.links['bus1'][link] == bus:
                    
                    d += elm.Line().color(bus_color).length(link_line_length) #Add line piece
                    d += (C := elm.Dot().color(link_color))
                    
                    n.links['end'][link] = C
            
            d += elm.Line(arrow = '-o').color(bus_color).length(link_line_length)
        
        
        w = ( (n.links.p_nom_opt/n.links.p_nom_opt.max()) )*4 + 0.1
        
        for link in n.links.index:
            # Loop through all links, and create lines with arrows.
            
            n.links.reindex(index1)
            
            w_link = w[link]
            
            style = 'N'
            
            d += ( elm.Wire(style, k = 1)
                  .color(link_color)
                  .at(n.links['start'][link].center)
                  .to(n.links['end'][link].center)
                  .label('p: ' + str(round(n.links.p_nom_opt[link],2)), 
                          fontsize = title_fontsize,
                          color = bus_color)
                  .zorder(0.1)
                  .linewidth(w_link)
                  )
            
            d += elm.Arrowhead(headwidth = headwidth, headlength = headlength).color(arrow_color)
            
            if handle_bi:
                # if link is bidirectional, add an additional arrow head.
                d += ( elm.Wire(style, k = 1)
                      .color(link_color)
                      .at(n.links['end'][link].center)
                      .to(n.links['start'][link].center)
                      .zorder(0)
                      .linewidth(w_link)
                      )
                
                d += elm.Arrowhead(headwidth = headwidth, headlength = headlength).color(arrow_color)
            
            