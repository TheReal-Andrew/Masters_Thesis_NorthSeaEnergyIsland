# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 21:19:36 2023

@author: lukas
"""

import schemdraw
import schemdraw.elements as elm
import os
import sys
# Add modules folder to path
sys.path.append(os.path.abspath('../../modules')) 

import pypsa_diagrams as pdiag

import pypsa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n = pypsa.Network()

n.madd('Bus',
       names = ['Island', 'Denmark', 'Norway', 'Germany', 'Belgium'])

n.madd('Generator',
       names         = ['Island Gen1', 'Island Gen2', 'Denmark Gen1', 'Norway Gen1'],
       bus           = ['Island',      'Island',      'Denmark',      'Norway'],
       p_nom         = [800,            700,          1000,            500],
       marginal_cost = [20,             10,           5,               25],
       )

n.madd('Load',
       names = ['Denmark Load1', 'Denmark Load2', 'Norway Load1', 'Belgium Load1'],
       bus   = ['Denmark',       'Denmark',       'Norway',       'Belgium'],
       p_set = [500,              200,            1400,           100],
       )

n.madd('Link',
       names = ['I-DK',    'DK-NO',   'NO-DK',  'DK-DE',   'DE-BE'],
       bus0  = ['Island',  'Denmark', 'Norway', 'Denmark', 'Germany'],
       bus1  = ['Denmark', 'Norway',  'Island', 'Germany', 'Belgium'],
       p_nom_extendable = [True, True, True, True, True],
       p_min_pu =         [-1,   0,    0,    0,    0],
       )

n.lopf(pyomo = False,
        solver_name = 'gurobi',
        )

#%%
# pdiag.draw_bus(n, 'Island')
# pdiag.draw_bus(n, 'Denmark')

pos = [
       [ 0,  0],
       [ 10,  -1],
       [ 8, 6],
       [8, -10],
        [0,  -11],
       ]

pdiag.draw_network(n, spacing = 1.5, 
                    pos = pos,
                   )

#%%

# with schemdraw.Drawing() as d2:
    
#     d2 += elm.Dot()
#     d2 += elm.Line()
    
#     d2  += elm.Arrowhead(headwidth = 1, headlength = 1)
    
#     d2 += elm.Dot()


# from schemdraw.segments import Segment, util, math, SegmentCircle

# # ----- Define custom elements -----
# class MyGen(elm.Element):
#     def __init__(self, *d, **kwargs):
#         super().__init__(*d, **kwargs)
#         self.segments.append(Segment(
#             [(0, 0), (0.75, 0)]))
#         sin_y = util.linspace(-.25, .25, num=25)
#         sin_x = [.2 * math.sin((sy-.25)*math.pi*2/.5) + 1.25 for sy in sin_y]
#         self.segments.append(Segment(list(zip(sin_x, sin_y))))
#         self.segments.append(SegmentCircle((1.25, 0), 0.5,))
        
# class MyLoad(elm.Element):
#     def __init__(self, *d, **kwargs):
#         super().__init__(*d, **kwargs)
#         lead = 0.95
#         h = 0.8
#         w = 0.5
#         self.segments.append(Segment(
#             [(0, 0), (0, lead), (-w, lead+h), (w, lead+h), (0, lead)]))
#         self.params['drop'] = (0, 0)
#         self.params['theta'] = 0
#         self.anchors['start'] = (0, 0)
#         self.anchors['center'] = (0, 0)
#         self.anchors['end'] = (0, 0)
        
# class MyStore(elm.Element):
#     def __init__(self, *d, **kwargs):
#         super().__init__(*d, **kwargs)
#         lead = 0.75
#         h = lead + 1
#         w = 1
#         self.segments.append(Segment(
#             [(0, 0), (lead, 0), (lead, w/2), (h, w/2),
#               (h, -w/2), (lead, -w/2), (lead, 0)
#               ]))
#%%

# bus_color  = 'steelblue'
# link_color = 'darkorange'
# fontsize  = 8
# title_fontsize = 12
# line_length = 1.5
# link_line_length = 0.75

#%%


# def draw_network(n, spacing = 2, line_length = 1.5, link_line_length = 0.75,
#                  fontsize = 8, title_fontsize = 12,
#                  bus_color = 'steelblue', link_color = 'darkorange'):

#     n_buses = len(n.buses.index)
#     T = [n_buses]
#     R = [n_buses * spacing]
    
#     # Create positions in a circle
#     pos = pdiag.circle_points(R, T)
#     s = pd.DataFrame(pos[0], columns = ['x', 'y'])
#     s.index = n.buses.index

#     # pdiag.plot_circle_points(R, T)

#     with schemdraw.Drawing() as d:
        
#         # Add columns with start and end cooridnates for links
#         n.links['start'] = np.nan
#         n.links['end']   = np.nan
        
#         for bus in n.buses.index:
#             d += (elm.Dot()
#                   .color(bus_color)
#                   .label(bus, fontsize = title_fontsize)
#                   .at(( s['x'][bus], s['y'][bus] )))
            
#             # ----- Get elements on this bus from network -----
#             gens   = n.generators[n.generators['bus'] == bus] #Get all generators on bus
#             loads  = n.loads[n.loads['bus'] == bus]
#             stores = n.stores[n.stores['bus'] == bus]
            
#             for link in n.links.index:
#                 # Loop through 
                
#                 if n.links['bus0'][link] == bus:
                    
#                     d += elm.Line().color(bus_color).length(link_line_length) #Add line piece
#                     d += (C := elm.Dot().color(link_color))
                    
#                     n.links['start'][link] = C
            
#             for gen in gens.index:
#                 # Loop through generators on this bus, and add new line segment
#                 # and generator icon. Add label to icon.
                
#                 d += elm.Line().color(bus_color).length(line_length) #Add line piece
#                 d.push() # Save position
#                 label = gen.replace(' ', ' \n') + '\n \n p: ' + str(round(n.generators.loc[gen].p_nom_opt, 2))
#                 d += MyGen().up().label(label, loc='right', fontsize = fontsize)
#                 d.pop()  # Return to saved position
            
#             for store in stores.index:
#                 # Loop through stores on this bus, and add new line segment
#                 # and store icon. Add label to icon.
                
#                 d += elm.Line().color(bus_color).length(line_length) #Add line piece
#                 d.push()
#                 label = store.replace(' ', ' \n') + '\n \n e: ' + str(round(n.stores.loc[store].e_nom_opt, 2))
#                 d += MyStore().up().label(label, loc = 'right', fontsize = fontsize)
#                 d.pop()
                
#             for load in loads.index:
#                 # Loop through loads on this bus, and add new line segment
#                 # and load icon. Add label to icon.
                
#                 d += elm.Line().color(bus_color).length(line_length) #Add line piece
#                 d.push()
#                 label = load.replace(' ', ' \n') + '\n \n mean p: ' + str(round(n.loads_t.p[load].mean(), 2))
#                 d += MyLoad().right().label(label, loc='top', fontsize = fontsize)
#                 d.pop()
                
#             for link in n.links.index:
                
#                 if n.links['bus1'][link] == bus:
                    
#                     d += elm.Line().color(bus_color).length(link_line_length) #Add line piece
#                     d += (C := elm.Dot().color(link_color))
                    
#                     n.links['end'][link] = C
            
#             d += elm.Line(arrow = '-o').color(bus_color).length(link_line_length)
        
#         for link in n.links.index:
            
#             d += ( elm.Wire('N', arrow = '->', k = 1, headwidth = 1)
#                   .color(link_color)
#                   .at(n.links['start'][link].center)
#                   .to(n.links['end'][link].center)
#                   .label('p: ' + str(n.links.p_nom_opt[link]), fontsize = fontsize)
#                   .zorder(0)
#                   )
            
# draw_network(n, spacing = 1.5)
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    