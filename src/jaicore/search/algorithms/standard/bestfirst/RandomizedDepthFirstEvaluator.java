package jaicore.search.algorithms.standard.bestfirst;

import java.util.Random;

import jaicore.search.algorithms.standard.core.NodeEvaluator;
import jaicore.search.structure.core.Node;

public class RandomizedDepthFirstEvaluator<T> implements NodeEvaluator<T,Integer> {

	private final Random rand;

	public RandomizedDepthFirstEvaluator(Random rand) {
		super();
		this.rand = rand;
	}

	@Override
	public Integer f(Node<T,Integer> node) {
		return (int) (-1 * (node.path().size() * 1000 + rand.nextInt(100)));
	}
}
