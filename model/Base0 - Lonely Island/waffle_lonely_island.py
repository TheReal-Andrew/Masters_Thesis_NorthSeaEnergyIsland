import os
import sys
# Add modules folder to path
os.chdir(os.path.join(os.path.dirname(__file__))) # Change working directory
sys.path.append(os.path.abspath('../../modules')) # Add modules to path
import matplotlib.pyplot as plt
import pypsa
import gorm as gm
import tim as tm
 
gm.set_plot_options()
year = 2030
#%% Import
 
# Choose one or more networks to import
n_2030     = pypsa.Network(f'v_{year}_base0_lonely_island_opt.nc')

# Choose one or more networks to loop through and insert data into
networks = [n_2030,
            # n_2030_nac,
            # n_2040,
            # n_2040_nac
            ]
 
# Insert necessary data into chosen networks
for n in networks:
    n.area_use      = tm.get_area_use()
    n.link_sum_max  = n.generators.p_nom_max['Wind']
    n.main_links    = n.links.loc[n.links.bus0 == "Energy Island"].index
 
 
#%% Make waffle diagrams
 
n = n_2030
 
#Can set title by passing a string to title parameter, title = 'yolo'
# title  = 'Title'  
 
# can set filename and save figure by passing filename to waffle function
# filename = f'waffle_{year}_area_capacity.pdf' 
 
gm.waffles_area_and_capacity(n,title = f"100% Renewable Energy Island in {year}")
plt.tight_layout() 
plt.savefig(f'../../images/green_island_{year}.pdf', format = 'pdf', bbox_inches='tight')