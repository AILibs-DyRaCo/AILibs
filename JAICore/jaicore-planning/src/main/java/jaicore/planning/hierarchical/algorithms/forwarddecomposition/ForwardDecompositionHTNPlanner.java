package jaicore.planning.hierarchical.algorithms.forwarddecomposition;

import jaicore.planning.core.Action;
import jaicore.planning.hierarchical.algorithms.GraphSearchBasedHTNPlanningAlgorithm;
import jaicore.planning.hierarchical.algorithms.forwarddecomposition.graphgenerators.tfd.TFDNode;
import jaicore.planning.hierarchical.problems.htn.IHTNPlanningProblem;
import jaicore.search.core.interfaces.IOptimalPathInORGraphSearchFactory;
import jaicore.search.probleminputs.GraphSearchInput;
import jaicore.search.probleminputs.builders.SearchProblemInputBuilder;

/**
 * Hierarchically create an object of type T
 * 
 * @author fmohr
 *
 * @param <T>
 */
public class ForwardDecompositionHTNPlanner<PA extends Action, IPlanning extends IHTNPlanningProblem, V extends Comparable<V>, ISearch extends GraphSearchInput<TFDNode, String>>
		extends GraphSearchBasedHTNPlanningAlgorithm<PA, IPlanning, ISearch, TFDNode, String, V> {

	public ForwardDecompositionHTNPlanner(IPlanning problem, IOptimalPathInORGraphSearchFactory<ISearch, TFDNode, String, V, ?, ?> searchFactory,
			SearchProblemInputBuilder<TFDNode, String, ISearch> searchProblemBuilder) {
		super(problem, new ForwardDecompositionReducer<PA, IPlanning>(), searchFactory, searchProblemBuilder);
	}
}
