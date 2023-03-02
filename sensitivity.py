import numpy as np
import matplotlib.pyplot as plt
import modules.island_plt as ip
ip.set_plot_options()

name = 'P2X'
step = 0.1
x = np.arange(0.5,1.5 + step, step)
# x = np.linspace(0.2,1.8,17)

sensitive_cost =  np.array([])
for i in x:
    
    cost1 = 13
    cost1 = cost1*i
    # cost2 = 12
    sensitive_cost = np.append(sensitive_cost, cost1)

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