import numpy as np
import matplotlib.pyplot as plt
import modules.island_plt as ip
import pypsa
import gorm as gm
import tim as tm
ip.set_plot_options()

name = 'P2X'
step = 0.1
x = np.arange(0.5,1.5 + step, step)
# x = np.linspace(0.2,1.8,17)

input_name = 'model/Base0/base0_opt.nc'
n = pypsa.Network(input_name)
n_opt = n.copy()

#%% Define custome constraints
def extra_functionalities(n, snapshots):
    gm.area_constraint(n, snapshots)
    gm.link_constraint(n, snapshots)

generator = ['P2X','Data']

sensitivity_cap = []
for i in generator:
    for i in np.arange(0.5,1.6, 0.1):
        n.generators.loc[generator, "marginal_cost"] = n_opt.generators.loc[generator, "marginal_cost"] * x
    
        #Solve
        n.lopf(pyomo = False,
               solver_name = 'gurobi',
               keep_shadowprices = True,
               keep_references = True,
               extra_functionality = extra_functionalities,
               )
        
        #Store installed capacities
        

#%% Plotting
fig, ax = plt.subplots()
ax.plot(x,sensitive_cost)

ax.set_title('Sensitivity of installed capacities with changing ' + name)
ax.set_xlabel('Change in cost of ' + name)
ax.set_ylabel('Installed capacities')

xtick_labels = [str(round((i-1)*100)) + "%" for i in x]
ax.set_xticks(x, xtick_labels)
ax.set_yticks(np.arange(0,20 + 2, 2))

ax.set_xlim([x.min(),x.max()])
ax.set_ylim([0,20])

plt.tight_layout() 
plt.savefig('images/sensitivity/' + name + '_sensitivity.pdf', format = 'pdf', bbox_inches='tight')