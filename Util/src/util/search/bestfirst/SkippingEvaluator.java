package util.search.bestfirst;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import util.search.core.NodeEvaluator;
import util.search.core.Node;

public class SkippingEvaluator<T,V extends Comparable<V>> implements NodeEvaluator<T,V> {

	private final NodeEvaluator<T,V> actualEvaluator;
	private final Random rand;
	private final float coin;
	private final Map<Node<T,V>, V> fCache = new HashMap<>();

	public SkippingEvaluator(NodeEvaluator<T,V> actualEvaluator, Random rand, float coin) {
		super();
		this.actualEvaluator = actualEvaluator;
		this.rand = rand;
		this.coin = coin;
	}

	@Override
	public V f(Node<T,V> node) {
		int depth = node.path().size() - 1;
		if (!fCache.containsKey(node)) {
			if (depth == 0) {
				fCache.put(node, actualEvaluator.f(node));
			} else {
				if (rand.nextFloat() >= coin) {
					fCache.put(node, actualEvaluator.f(node));
				} else {
					fCache.put(node, f(node.getParent()));
				}
			}
		}
		return fCache.get(node);
	}

	@Override
	public String toString() {
		return "SkippingEvaluator [actualEvaluator=" + actualEvaluator + "]";
	}
}