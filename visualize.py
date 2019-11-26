import networkx as nx

import matplotlib.pyplot as plt

from neat.phenotypes import Phenotype
from neat.utils import Singleton

# class Visualize():

# 	def __init__(self) -> None:
# 		self.queue = rpyc.connect("localhost", 18811)

# 	def update(self, neuralNetwork: Phenotype):
# 		self.neuralNetwork = neuralNetwork

# 		edges = {"x": [], "y": []}
# 		for edge in [e for n in self.neuralNetwork.neurons for e in n.linksIn]:
# 			edges['x'].append(tuple([edge.fromNeuron.x, edge.toNeuron.x, None]))
# 			edges['y'].append(tuple([edge.fromNeuron.y, edge.toNeuron.y, None]))
		
# 		# print(edges)
		
# 		neurons = {"x": [], "y": []}
# 		for neuron in self.neuralNetwork.neurons:
# 			neurons['x'].append(neuron.x)
# 			neurons['y'].append(neuron.y)

# 		print("Visualize -> neurons: {}".format(len(neurons)))

# 		dataToSend = json.dumps([neurons, edges])
# 		self.queue.root.send_message(dataToSend)


class Visualize(metaclass=Singleton):

	def __init__(self):
		pass
		# self.fig = plt.figure(figsize=(1,2))
		# self.fig, self.axs = plt.subplots(2, 2)

		# self.top_left = self.axs[0, 0]
		# self.top_left.set_title("")
		#
		# self.top_right = self.axs[0, 1]
		# self.top_right.set_title("")
		#
		# self.bottom_left = self.axs[1, 0]
		# self.bottom_left.set_title("")
		#
		# self.bottom_right = self.axs[1, 1]
		# self.bottom_right.set_title("")

		# plt.tight_layout()
		# plt.ion()
		# plt.show()
		# plt.draw()
		# plt.pause(0.1)

	def update(self, phenotype: Phenotype):
		G = phenotype.graph
		# print(G.nodes())
		fig, ax = plt.subplots()
		pos = nx.get_node_attributes(G, 'pos')
		# print(pos)

		nx.draw_networkx_nodes(G, pos, ax=ax)
		nx.draw_networkx_edges(G, pos, ax=ax, edge_vmin=-1.0, edge_vmax=1.0)
		ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

		# print(nx.adjacency_matrix(G).todense())

		plt.show()
		# visualize(G)

	# def close(self):
	# 	plt.close()
		

