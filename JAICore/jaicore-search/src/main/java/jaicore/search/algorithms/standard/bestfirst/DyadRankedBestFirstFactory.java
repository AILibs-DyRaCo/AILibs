package jaicore.search.algorithms.standard.bestfirst;

import jaicore.search.model.travesaltree.ReducedGraphGenerator;
import jaicore.search.probleminputs.GraphSearchWithSubpathEvaluationsInput;

public class DyadRankedBestFirstFactory<N, A, V extends Comparable<V>> extends StandardBestFirstFactory<N, A, V> {

	private IBestFirstQueueConfiguration<GraphSearchWithSubpathEvaluationsInput<N, A, V>, N, A, V> OPENConfig;

	public DyadRankedBestFirstFactory(
			IBestFirstQueueConfiguration<GraphSearchWithSubpathEvaluationsInput<N, A, V>, N, A, V> OPENConfig) {
		this.OPENConfig = OPENConfig;
	}

	@Override
	public BestFirst<GraphSearchWithSubpathEvaluationsInput<N, A, V>, N, A, V> getAlgorithm() {
		// Replace graph generator in problem
		this.setProblemInput(new GraphSearchWithSubpathEvaluationsInput<>(
				new ReducedGraphGenerator<>(this.getInput().getGraphGenerator()), this.getInput().getNodeEvaluator()));

		// Configure and return best first
		BestFirst<GraphSearchWithSubpathEvaluationsInput<N, A, V>, N, A, V> bestFirst = super.getAlgorithm();
		OPENConfig.configureBestFirst(bestFirst);
		return bestFirst;
	}
}
