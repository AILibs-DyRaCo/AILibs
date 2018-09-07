package jaicore.search.problemtransformers;

import java.util.Random;

import jaicore.basic.algorithm.AlgorithmProblemTransformer;
import jaicore.search.algorithms.standard.bestfirst.nodeevaluation.AlternativeNodeEvaluator;
import jaicore.search.algorithms.standard.bestfirst.nodeevaluation.INodeEvaluator;
import jaicore.search.algorithms.standard.bestfirst.nodeevaluation.RandomCompletionBasedNodeEvaluator;
import jaicore.search.model.probleminputs.GeneralEvaluatedTraversalTree;
import jaicore.search.model.probleminputs.GraphSearchProblemInput;

public class GraphSearchProblemInputToGeneralEvaluatedTraversalTreeViaRDFS<N, A, V extends Comparable<V>>
		implements AlgorithmProblemTransformer<GraphSearchProblemInput<N, A, V>, GeneralEvaluatedTraversalTree<N, A, V>> {

	private final INodeEvaluator<N, V> preferredNodeEvaluator;
	private final INodeEvaluator<N, Double> preferredNodeEvaluatorForRandomCompletion;
	private final int seed;
	private final int numSamples;
	private final int timeoutForSingleCompletionEvaluationInMS;
	private final int timeoutForNodeEvaluationInMS;

	public GraphSearchProblemInputToGeneralEvaluatedTraversalTreeViaRDFS(INodeEvaluator<N, V> preferredNodeEvaluator, INodeEvaluator<N, Double> preferredNodeEvaluatorForRandomCompletion, int seed, int numSamples, int timeoutForSingleCompletionEvaluationInMS, int timeoutForNodeEvaluationInMS) {
		super();
		this.preferredNodeEvaluator = preferredNodeEvaluator;
		this.preferredNodeEvaluatorForRandomCompletion = preferredNodeEvaluatorForRandomCompletion;
		this.seed = seed;
		this.numSamples = numSamples;
		this.timeoutForSingleCompletionEvaluationInMS = timeoutForSingleCompletionEvaluationInMS;
		this.timeoutForNodeEvaluationInMS = timeoutForNodeEvaluationInMS;
	}

	public INodeEvaluator<N, V> getPreferredNodeEvaluator() {
		return preferredNodeEvaluator;
	}

	public INodeEvaluator<N, Double> getPreferredNodeEvaluatorForRandomCompletion() {
		return preferredNodeEvaluatorForRandomCompletion;
	}

	public int getSeed() {
		return seed;
	}

	public int getNumSamples() {
		return numSamples;
	}

	@Override
	public GeneralEvaluatedTraversalTree<N, A, V> transform(GraphSearchProblemInput<N, A, V> problem) {
		RandomCompletionBasedNodeEvaluator<N, V> rc = new RandomCompletionBasedNodeEvaluator<>(new Random(seed), numSamples, problem.getPathEvaluator(), timeoutForSingleCompletionEvaluationInMS, timeoutForNodeEvaluationInMS, preferredNodeEvaluatorForRandomCompletion);
		return new GeneralEvaluatedTraversalTree<>(problem.getGraphGenerator(), new AlternativeNodeEvaluator<>(preferredNodeEvaluator, rc));
	}

}
