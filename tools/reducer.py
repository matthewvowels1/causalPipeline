
import networkx as nx

def project_causes_func(xs, ys, ordering, all_causally_relevant_vars, reduced_graph):
	to_remove = []
	ordered_causes = ([x for x in ordering if x in all_causally_relevant_vars])
	for cause1 in ordered_causes:
		cause1_children = list(reduced_graph.successors(cause1))
		for cause2 in ordered_causes[1:]:
			if (cause2 in cause1_children):
				cause2_rents = set(list(reduced_graph.predecessors(cause2)))
				cause2_rents = cause2_rents - set([cause1])
				if len(cause2_rents) == 0:
					cause2_children = list(reduced_graph.successors(cause2))
					to_remove.append(cause2)
					for child in cause2_children:
						reduced_graph.add_edge(cause1, child)
	for remove in to_remove:
		if remove not in xs and remove not in ys:
			reduced_graph.remove_node(remove)
	return reduced_graph


def project_confs_func(all_confounders, reduced_graph, remaining_nodes):
	to_remove = []
	for conf_node in all_confounders:
		conf_node_decs = list(nx.descendants(reduced_graph, conf_node))
		for remaining_node in remaining_nodes:
			if remaining_node in conf_node_decs:
				remaining_parents = set(list(reduced_graph.predecessors(remaining_node)))
				remaining_parents = remaining_parents - set([conf_node])
				if len(remaining_parents) == 0:
					remaining_children = list(reduced_graph.successors(remaining_node))
					to_remove.append(remaining_node)
					for child in remaining_children:
						reduced_graph.add_edge(conf_node, child)

	for remove in to_remove:
		reduced_graph.remove_node(remove)
	return reduced_graph


def precision_func(remaining_nodes, xs, ys, reduced_graph, all_confounders, remove_precision):
	precision_nodes = []
	for remaining_node in remaining_nodes:
		# check ancestors and descendents
		remaining_ancs = list(nx.ancestors(reduced_graph, remaining_node))
		if any(x in remaining_ancs for x in all_confounders) == False:
			if (remaining_node not in xs) and (remaining_node not in ys):
				if remove_precision:
					reduced_graph.remove_node(remaining_node)
				precision_nodes.append(remaining_node)
	precision_nodes = set(precision_nodes)
	return precision_nodes, reduced_graph


def remove_confounders(xs, ordering, causal_chain, reduced_graph, all_causally_relevant_vars, project_causes):
	# this code removes confounders <and> instruments! If project_causes then stop after first cause
	# init confounder dictionary (based on causal paths from root causes)
	confounders_dict = {}
	k = 0
	all_confounders = []
	some_left_flag = True
	while some_left_flag:

		for x in xs:
			effect_vars = list(nx.descendants(reduced_graph, x))
			try:
				current_cause = causal_chain[x][k]
			except:
				some_left_flag = False

			if k < 1:
				confounders_dict[x] = []

			conf_vars_ = list(nx.ancestors(reduced_graph, current_cause))
			conf_vars = []
			[conf_vars.append(v) for v in conf_vars_ if v not in all_causally_relevant_vars]
			for potential_confounder in conf_vars:
				pot_conf_descs = list(nx.descendants(reduced_graph, potential_confounder))
				if any(p in pot_conf_descs for p in effect_vars):

					potential_confounder_ancs = list(nx.ancestors(reduced_graph, potential_confounder))
					if not any(p in potential_confounder_ancs for p in all_confounders):
						if (potential_confounder not in confounders_dict[x]) and (
								potential_confounder not in all_causally_relevant_vars):
							conf_children = list(reduced_graph.successors(potential_confounder))
							if current_cause in conf_children:
								reduced_graph.remove_edge(potential_confounder, current_cause)
								confounders_dict[x].append(potential_confounder)
								all_confounders.append(potential_confounder)

		if project_causes:
			break
		else:
			unordered_children = list(reduced_graph.successors(current_cause))
			children = []
			for var in ordering:
				if var in unordered_children:
					children.append(var)

			if len(children) > 0:
				causal_chain[x].extend(children)
			k += 1

	return all_confounders, reduced_graph, confounders_dict


def get_causal_vars(xs, ys, reduced_graph):
	# init causally relevant variable dictionary
	all_causally_relevant_vars = []
	all_causally_relevant_vars.extend(xs)
	all_causally_relevant_vars.extend(ys)
	# init causal chain dictionary - this will keep track of children based on root causes
	causal_chain = {}
	for x in xs:
		causal_chain[x] = [x]
		effect_vars = list(nx.descendants(reduced_graph, x))
		# add causally relevant vars to dict of causally relevant vars
		all_causally_relevant_vars.extend(effect_vars)
		all_causally_relevant_vars = list(set(all_causally_relevant_vars))
	return causal_chain, all_causally_relevant_vars


def reducer(graph, xs, ys, remove_precision=True, project_confs=True, project_causes=True):
	if not project_causes:
		print('WARNING: If including mediating paths then some unneeded (but otherwise benign) confounding paths to outcome may remain.')

	has_path = 0
	for x in xs:
		for y in ys:
			hp = nx.has_path(graph, x, y)
			if hp:
				has_path += 1

	if has_path == 0:
		print('WARNING: No causal paths between xs and ys, introducing direct path!')
		graph.add_edge(xs[0], ys[0])

	reduced_graph = graph.copy()
	reduced_graph.remove_nodes_from(list(nx.isolates(reduced_graph)))

	##### Remove nodes <after> ys: #####
	ordering = list(nx.topological_sort(reduced_graph))
	indexes = []
	for var in ys:
		indexes.append(ordering.index(var))
	max_index = max(indexes)
	remove_vars = ordering[max_index + 1:]
	for var in remove_vars:
		reduced_graph.remove_node(var)

	ordering = list(nx.topological_sort(reduced_graph))



	causal_chain, all_causally_relevant_vars = get_causal_vars(xs, ys, reduced_graph)
	# get list of all nodes
	all_nodes = set(list(reduced_graph.nodes))
	# identify list of non-causal nodes
	non_causal_nodes = all_nodes - set(all_causally_relevant_vars)

	##### PROJECT CAUSAL PATHS ######
	if project_causes:
		reduced_graph = project_causes_func(xs, ys, ordering, all_causally_relevant_vars, reduced_graph)

	##### REMOVE CONFOUNDING PATHS ######
	all_confounders, reduced_graph, confounders_dict = remove_confounders(xs, ordering, causal_chain, reduced_graph,
	                                                    all_causally_relevant_vars, project_causes)
	all_confounders = set(all_confounders)
	# find remaining nodes which are neither causal nor confounders (e.g. precisions and colliders)
	remaining_nodes = non_causal_nodes - all_confounders

	##### REMOVE PRECISION VARS ######
	precision_nodes, reduced_graph = precision_func(remaining_nodes, xs, ys, reduced_graph, all_confounders, remove_precision)
	remaining_nodes = remaining_nodes - precision_nodes

	##### PROJECT CONFOUNDING PATHS ######
	if project_confs:
		reduced_graph = project_confs_func(all_confounders, reduced_graph, remaining_nodes)

	##### PROJECT CAUSAL AGAIN ######
	if project_causes:  # run this again
		ordering = list(nx.topological_sort(reduced_graph))
		reduced_graph = project_causes_func(xs, ys, ordering, all_causally_relevant_vars, reduced_graph)

	# finally clean up graph by removing isolated vars
	reduced_graph.remove_nodes_from(list(nx.isolates(reduced_graph)))

	if len(xs) > 1:
		print('WARNING: List of confounders is strictly valid for single cause graphs.')
	all_confounders = set(all_confounders)
	all_nodes = set(list(reduced_graph.nodes()))


	return reduced_graph, all_confounders.intersection(all_nodes), precision_nodes
