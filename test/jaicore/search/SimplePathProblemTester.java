package jaicore.search;

import java.util.ArrayList;
import java.util.Collection;

import org.junit.Test;

import jaicore.graphvisualizer.SimpleGraphVisualizationWindow;
import jaicore.search.algorithms.standard.bestfirst.BestFirst;
import jaicore.search.structure.core.GraphGenerator;
import jaicore.search.structure.core.NodeExpansionDescription;
import jaicore.search.structure.core.NodeType;
import jaicore.search.structure.graphgenerator.PathGoalTester;
import jaicore.search.structure.graphgenerator.SingleRootGenerator;
import jaicore.search.structure.graphgenerator.SuccessorGenerator;

public class SimplePathProblemTester {

	@Test
	public void test() {
		
		GraphGenerator<Integer, Object> gen = new GraphGenerator<Integer,Object>() {

			@Override
			public SingleRootGenerator<Integer> getRootGenerator() {
				return () -> 0;
			}

			@Override
			public SuccessorGenerator<Integer, Object> getSuccessorGenerator() {
				return n -> {
					Collection<NodeExpansionDescription<Integer, Object>> succ = new ArrayList<>();
					succ.add(new NodeExpansionDescription<Integer, Object>(n, n * 2, null, NodeType.OR));
					succ.add(new NodeExpansionDescription<Integer, Object>(n, (n + 1) * 2, null, NodeType.OR));
					return succ;
				};
			}

			@Override
			public PathGoalTester<Integer> getGoalTester() {
				return p -> {
					int sum = 0; 
					for (Integer i : p) {
						sum += i;
					}
					return sum > 0 && (sum % 100) == 0;
				};
			}
		};
		
		BestFirst<Integer, Object> bf = new BestFirst<>(gen, n -> 0);
		new SimpleGraphVisualizationWindow<>(bf.getEventBus()).getPanel().setTooltipGenerator(n -> n.getPoint().toString());
		bf.nextSolution();
		while (true);
	}
}
