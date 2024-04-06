#!/usr/bin/python3

# Copyright 2023, 2024 Philipp Andelfinger, Justin Kreikemeyer
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#  
#   The above copyright notice and this permission notice shall be included in all copies or
#   substantial portions of the Software.
#   
#   THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#   PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
#   ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
#   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

# generates a random network.
# args: number of nodes:

import sys 
import networkx as nx
import random
import matplotlib.pyplot as plt

random.seed(int(sys.argv[1]))

plot = False
plot_fname = "network.pdf"
network_file = "network.txt"

num_nodes = int(sys.argv[2])

if num_nodes == 10:
  G = nx.random_geometric_graph(num_nodes, 0.4)
elif num_nodes == 25:
  G = nx.random_geometric_graph(num_nodes, 0.32)
elif num_nodes == 50:
  G = nx.random_geometric_graph(num_nodes, 0.203)
elif num_nodes == 100:
  G = nx.random_geometric_graph(num_nodes, 0.16)
elif num_nodes == 500:
  G = nx.random_geometric_graph(num_nodes, 0.0575)
elif num_nodes == 5000:
  G = nx.random_geometric_graph(num_nodes, 0.018)
else:
  print("unknown network size")
  sys.exit(1)

nx.draw_networkx(G)
plt.show()
plt.savefig(plot_fname)

print("avg. degree:", sum((x[1] for x in G.degree())) / G.number_of_nodes())
print("max. degree:", max(x[1] for x in G.degree()))

with open(network_file, "w") as f:
  for n0, n1 in G.edges():
    f.write(str(n0) + " " + str(n1) + "\n")

