from itertools import combinations, permutations, chain
import networkx as nx
from pgmpy.base import PDAG
from pgmpy.estimators import StructureEstimator
from pgmpy.estimators.CITests import chi_square, pearsonr, independence_match
from pgmpy.global_vars import SHOW_PROGRESS
from mi_test_func_pgmpy import*
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import logging


class PC_adapted(StructureEstimator):
    def __init__(self, data=None, independencies=None, **kwargs):
        """
        Class for constraint-based estimation of DAGs using the PC algorithm
        from a given data set.  Identifies (conditional) dependencies in data
        set using chi_square dependency test and uses the PC algorithm to
        estimate a DAG pattern that satisfies the identified dependencies. The
        DAG pattern can then be completed to a faithful DAG, if possible.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.  (If some
            values in the data are missing the data cells should be set to
            `numpy.NaN`.  Note that pandas converts each column containing
            `numpy.NaN`s to dtype `float`.)

        References
        ----------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques,
            2009, Section 18.2
        [2] Neapolitan, Learning Bayesian Networks, Section 10.1.2 for the PC algorithm (page 550), http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        """
        super(PC_adapted, self).__init__(data=data, independencies=independencies, **kwargs)

    def estimate(
        self,
        ci_test="chi_square",
        max_cond_vars=5,
        return_type="dag",
        significance_level=0.01,
        n_jobs=-1,
        knn=5,
        show_progress=True
    ):

        # Step 0: Do checks that the specified parameters are correct, else throw meaningful error.
        if (not callable(ci_test)) and (
            ci_test not in ("chi_square", "independence_match", "gcit", "pearsonr", "mixed_mi", "cont_mi")
        ):
            raise ValueError(
                "ci_test must be a callable or one of: chi_square, pearsonr, independence_match"
            )

        if (ci_test == "independence_match") and (self.independencies is None):
            raise ValueError(
                "For using independence_match, independencies argument must be specified"
            )
        elif (ci_test in ("chi_square", "pearsonr")) and (self.data is None):
            raise ValueError(
                "For using Chi Square or Pearsonr, data arguement must be specified"
            )

        # Step 1: Run the PC algorithm to build the skeleton and get the separating sets.
        skel, separating_sets = self.build_skeleton(
            ci_test=ci_test,
            max_cond_vars=max_cond_vars,
            significance_level=significance_level,
            show_progress=show_progress,
            knn=knn
        )

        if return_type.lower() == "skeleton":
            return skel, separating_sets

        # Step 2: Orient the edges based on build the PDAG/CPDAG.
        pdag = self.skeleton_to_pdag(skel, separating_sets)

        # Step 3: Either return the CPDAG or fully orient the edges to build a DAG.
        if return_type.lower() in ("pdag", "cpdag"):
            return pdag
        elif return_type.lower() == "dag":
            return pdag.to_dag()
        else:
            raise ValueError(
                f"return_type must be one of: dag, pdag, cpdag, or skeleton. Got: {return_type}"
            )


    def build_skeleton(
        self,
        ci_test="chi_square",
        max_cond_vars=5,
        significance_level=0.01,
        show_progress=True,
        knn=5
    ):

        # Initialize initial values and structures.
        lim_neighbors = 0
        separating_sets = dict()
        if ci_test == "chi_square":
            ci_test = chi_square
        elif ci_test == "pearsonr":
            ci_test = pearsonr
        elif ci_test == "mixed_mi":
            ci_test = mixed_cmi
        elif ci_test == "gcit":
            ci_test = gc_it
        elif ci_test == "cont_mi":
            ci_test = ksg_cmi
        elif ci_test == "independence_match":
            ci_test = independence_match
        elif callable(ci_test):
            ci_test = ci_test
        else:
            raise ValueError(
                f"ci_test must either be chi_square, pearsonr, gcit (GAN), independence_match, mixed_mi, or cont_mi, or a function. Got: {ci_test}"
            )

        if show_progress and SHOW_PROGRESS:
            pbar = tqdm(total=max_cond_vars)
            pbar.set_description("Working for n conditional variables: 0")

        # Step 1: Initialize a fully connected undirected graph
        graph = nx.complete_graph(n=self.variables, create_using=nx.Graph)

        # Exit condition: 1. If all the nodes in graph has less than `lim_neighbors` neighbors.
        #             or  2. `lim_neighbors` is greater than `max_conditional_variables`.
        while not all(
            [len(list(graph.neighbors(var))) < lim_neighbors for var in self.variables]
        ):

            # Step 2: Iterate over the edges and find a conditioning set of
            # size `lim_neighbors` which makes u and v independent.
            u_prev, v_prev, sepset_prev = None, None, None  # prevents repeating the same eval twice
            # In case of stable, precompute neighbors as this is the stable algorithm.
            neighbors = {node: set(graph[node]) for node in graph.nodes()}
            for (u, v) in graph.edges():
                for separating_set in chain(
                    combinations(set(graph.neighbors(u)) - set([v]), lim_neighbors),
                    combinations(set(graph.neighbors(v)) - set([u]), lim_neighbors),
                ):
                    # If a conditioning set exists remove the edge, store the
                    # separating set and move on to finding conditioning set for next edge.
                    if u != u_prev or v != v_prev or sepset_prev != sepset_prev:
                        val = ci_test(
                            X=u,
                            Y=v,
                            Z=separating_set,
                            data=self.data,
                            significance_level=significance_level,
                            knn=knn)
                        if val:
                            print(val, u, v, separating_set)
                            separating_sets[frozenset((u, v))] = separating_set
                            graph.remove_edge(u, v)
                            u_prev, v_prev, sepset_prev = u, v, separating_set
                            break
                        else:
                            print(val, u, v, separating_set)
                            u_prev, v_prev, sepset_prev = u, v, separating_set

            # Step 3: After iterating over all the edges, expand the search space by increasing the size
            #         of conditioning set by 1.
            if lim_neighbors >= max_cond_vars:
                logging.info(
                    "Reached maximum number of allowed conditional variables. Exiting"
                )
                break
            lim_neighbors += 1

            if show_progress and SHOW_PROGRESS:
                pbar.update(1)
                pbar.set_description(
                    f"Working for n conditional variables: {lim_neighbors}"
                )

        if show_progress and SHOW_PROGRESS:
            pbar.close()
        return graph, separating_sets


    @staticmethod
    def skeleton_to_pdag(skeleton, separating_sets):

        pdag = skeleton.to_directed()
        node_pairs = list(permutations(pdag.nodes(), 2))

        # 1) for each X-Z-Y, if Z not in the separating set of X,Y, then orient edges as X->Z<-Y
        # (Algorithm 3.4 in Koller & Friedman PGM, page 86)
        for pair in node_pairs:
            X, Y = pair
            if not skeleton.has_edge(X, Y):
                for Z in set(skeleton.neighbors(X)) & set(skeleton.neighbors(Y)):
                    if Z not in separating_sets[frozenset((X, Y))]:
                        pdag.remove_edges_from([(Z, X), (Z, Y)])

        progress = True
        while progress:  # as long as edges can be oriented (removed)
            num_edges = pdag.number_of_edges()

            # 2) for each X->Z-Y, orient edges to Z->Y
            # (Explanation in Koller & Friedman PGM, page 88)
            for pair in node_pairs:
                X, Y = pair
                if not pdag.has_edge(X, Y):
                    for Z in (set(pdag.successors(X)) - set(pdag.predecessors(X))) & (
                        set(pdag.successors(Y)) & set(pdag.predecessors(Y))
                    ):
                        pdag.remove_edge(Y, Z)

            # 3) for each X-Y with a directed path from X to Y, orient edges to X->Y
            for pair in node_pairs:
                X, Y = pair
                if pdag.has_edge(Y, X) and pdag.has_edge(X, Y):
                    for path in nx.all_simple_paths(pdag, X, Y):
                        is_directed = True
                        for src, dst in list(zip(path, path[1:])):
                            if pdag.has_edge(dst, src):
                                is_directed = False
                        if is_directed:
                            pdag.remove_edge(Y, X)
                            break

            # 4) for each X-Z-Y with X->W, Y->W, and Z-W, orient edges to Z->W
            for pair in node_pairs:
                X, Y = pair
                for Z in (
                    set(pdag.successors(X))
                    & set(pdag.predecessors(X))
                    & set(pdag.successors(Y))
                    & set(pdag.predecessors(Y))
                ):
                    for W in (
                        (set(pdag.successors(X)) - set(pdag.predecessors(X)))
                        & (set(pdag.successors(Y)) - set(pdag.predecessors(Y)))
                        & (set(pdag.successors(Z)) & set(pdag.predecessors(Z)))
                    ):
                        pdag.remove_edge(W, Z)

            progress = num_edges > pdag.number_of_edges()

        # TODO: This is temp fix to get a PDAG object.
        edges = set(pdag.edges())
        undirected_edges = []
        directed_edges = []
        for u, v in edges:
            if (v, u) in edges:
                undirected_edges.append((u, v))
            else:
                directed_edges.append((u, v))
        return PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)