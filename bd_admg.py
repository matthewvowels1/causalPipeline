import numpy as np
import networkx as nx
import itertools

def check_bd_id(graph, X, Y):
	'''
	:param graph: networkx DiGraph()
	:param X: name of cause node
	:param Y: name of effect node
	:return: tuple(boolean, list) whether backdoor criterion fulfilled and list of sufficient backdoor adjustment nodes
	'''

	t_preds = list(graph.predecessors(X))

	mod_graph = graph.copy()
	mod_graph.remove_node(X)

	y_preds = list(nx.ancestors(mod_graph, Y))

	bd_adjust_set = list(set(t_preds) & set(y_preds))

	flag = 0
	for var in bd_adjust_set:
		if 'U' in str(var):
			flag = 1

	if flag == 0:
		return True, bd_adjust_set
	else:
		return False, None



def bd_brute(graph, X, Y):
	'''

	:param graph: probabilistic networkx DiGraph() (with weights [0.5,1])
	:param X:  name of cause node
	:param Y:  name of effect node
	:return:
	'''
	check, _ = check_bd_id(graph=graph, X=X, Y=Y)
	E = set()
	if check:
		return E, 0

	else:
		# get all node names
		nodes = list(graph.nodes())
		Unodes = np.array([node for node in nodes if 'U' in str(node)])
		Ucosts = []
		for Unode in Unodes:
			edges = list(graph.edges(Unode, data=True))
			Ucosts.append(edges[0][2]['weight'])
		Ucosts = np.array(Ucosts)

		Unodes_sorted = Unodes[np.argsort(Ucosts)]
		Ucosts_sorted = Ucosts[np.argsort(Ucosts)]
		Unode_index = np.arange(0, len(Unodes_sorted))

		best_cost = np.inf

		j = 1
		iter_num = 0
		while True:
			combs = list(itertools.combinations(Unode_index, j))
			combs_costs = []
			for comb in combs:
				cost = Ucosts_sorted[list(comb)].sum()
				combs_costs.append(cost)

			# sort costs and combinations by cost
			order_inds = np.argsort(combs_costs)
			combs_costs = np.asarray(combs_costs)[order_inds]
			combs = np.asarray(combs)[order_inds]

			k = 0

			while True:
				comb = list(combs[k])
				current_cost = combs_costs[k]

				trial_graph = graph.copy()
				node_names = Unodes_sorted[comb]
				for comb_node in node_names:
					trial_graph.remove_node(comb_node)

				check, _ = check_bd_id(graph=trial_graph, X=X, Y=Y)
				if check:
					if current_cost < best_cost:
						best_cost = current_cost
						E = set(node_names)
				k += 1

				if k >= len(combs):
					break
			j += 1
			if j >= len(Unode_index):
				break
	return E, best_cost


def get_probs(cutset, graph):
	edges = graph.edges(data=True)
	log_probs = []
	names = []
	for edge in edges:
		name = edge[0]
		if 'U' in str(name):
			if name not in names:
				prob = edge[2]['weight']
				log_probs.append(np.log(prob))
		else:
			prob = edge[2]['weight']
			log_probs.append(np.log(prob))

		names.append(name)
	before_log_sum = np.sum(log_probs)  # this is the sum(log probs) of the original graph

	graph_ = graph.copy()

	cutset_probs = [] # these are the log probabilities of the remove edges
	for cutnode in cutset:
		cost = list(graph.edges(cutnode, data=True))[0][2]['weight']
		graph_.remove_node(cutnode)
		cutset_probs.append(cost)

	edges = graph_.edges(data=True)
	log_probs = []
	names = []
	for edge in edges:
		name = edge[0]
		if 'U' in str(name):
			if name not in names:
				prob = edge[2]['weight']
				log_probs.append(np.log(prob))
		else:
			prob = edge[2]['weight']
			log_probs.append(np.log(prob))

		names.append(name)
	after_log_sum = np.sum(log_probs)   # this is the sum(log probs) of the reduced graph

	return before_log_sum, after_log_sum, (np.log(1 - np.asarray(cutset_probs))).sum()




if __name__ == '__main__':

	graph = nx.DiGraph()
	graph.add_edge(0, 1, weight=1.0)
	graph.add_edge(1, 2, weight=1.0)
	graph.add_edge(2, 3, weight=1.0)
	graph.add_edge(4, 0, weight=1.0)
	graph.add_edge(3, 5, weight=1.0)
	graph.add_edge(4, 1, weight=1.0)
	graph.add_edge(6, 0, weight=1.0)
	graph.add_edge(6, 7, weight=1.0)
	graph.add_edge(7, 3, weight=1.0)
	graph.add_edge(0, 8, weight=1.0)
	graph.add_edge(3, 8, weight=1.0)
	graph.add_edge(8, 9, weight=1.0)
	graph.add_edge('U0', 0, weight=0.9)
	graph.add_edge('U0', 1, weight=0.9)
	graph.add_edge('U1', 0, weight=0.51)
	graph.add_edge('U1', 3, weight=0.51)
	graph.add_edge('U2', 8, weight=1.0)
	graph.add_edge('U2', 3, weight=1.0)
	graph.add_edge(10, 0, weight=1.0)

	# TEST BACKDOOR ID
	check, bd_set = check_bd_id(graph=graph, X=0, Y=3)

	if check:
		print('Is identifiable with backdoor set:', bd_set)
	else:
		print('Backdoor criterion not fulfilled, finding confounders to remove...')
		# RUNNING BRUTE FORCE ALGO TO FIND SOLUTION AND COST
		E, best_cost = bd_brute(graph=graph, X=0, Y=3)

		if best_cost != np.inf:
			print('Identifiable with the removal of:', E, ' at a cost of:', best_cost)
		else:
			print('No solution.')

		if best_cost != 0:
			# FINDING PLAUSIBILITY RATIO OF SOLUTION TO ORIGINAL GRAPH
			before_logsum, after_logsum, cutset_invlogsum = get_probs(E, graph)
			ratio = np.exp((after_logsum + cutset_invlogsum)) / np.exp(before_logsum)
			print('Plausibility ratio:', ratio)
		else:
			ratio = 1.0

		# REMOVE NODES FROM GRAPH AND TEST BACKDOOR ID AGAIN, PROVIDING SUFFICIENT ADJUSTMENT SET
		for node in E:
			graph.remove_node(node)

		check, bd_set = check_bd_id(graph=graph, X=0, Y=3)

		if check:
			print('Is identifiable with backdoor set:', bd_set)
		else:
			print('Backdoor criterion not fulfilled.')