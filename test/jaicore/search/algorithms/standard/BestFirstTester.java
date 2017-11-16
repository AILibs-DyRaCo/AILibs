package jaicore.search.algorithms.standard;

import static org.junit.Assert.assertNotNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import jaicore.basic.PerformanceLogger;
import jaicore.basic.PerformanceLogger.PerformanceMeasure;
import jaicore.graphvisualizer.SimpleGraphVisualizationWindow;
import jaicore.search.algorithms.standard.bestfirst.BestFirst;
import jaicore.search.structure.core.GraphGenerator;
import jaicore.search.structure.core.NodeExpansionDescription;
import jaicore.search.structure.core.NodeType;
import jaicore.search.structure.graphgenerator.NodeGoalTester;
import jaicore.search.structure.graphgenerator.PathGoalTester;
import jaicore.search.structure.graphgenerator.RootGenerator;
import jaicore.search.structure.graphgenerator.SuccessorGenerator;

public class BestFirstTester {

	static class TestNode {
		static int size = 0;
		int value = size++;
		
		public String toString() { return "" + value; }
	}

	@Test
	public void test() {
		
		GraphGenerator<TestNode, String> gen = new GraphGenerator<BestFirstTester.TestNode, String>() {

			@Override
			public RootGenerator<TestNode> getRootGenerator() {
				return () -> Arrays.asList(new TestNode[]{new TestNode()});
			}

			@Override
			public SuccessorGenerator<TestNode, String> getSuccessorGenerator() {
				return n -> {
					List<NodeExpansionDescription<TestNode,String>> l = new ArrayList<>(3);
					for (int i = 0; i < 3; i++) {
						l.add(new NodeExpansionDescription<>(n, new TestNode(), "edge label", NodeType.OR));
					}
					return l;
				};
			}

			/*
			 * only one of the getGoalTester-Methods has to be implemented
			 */
			@Override
			public PathGoalTester<TestNode> getPathGoalTester() {
				return l -> l.get(l.size()-1).value == 1000;
			}

			@Override
			public NodeGoalTester<TestNode> getNodeGoalTester() {
				return null;
			}
			
			
		};
		
		BestFirst<TestNode,String> bf = new BestFirst<>(gen, n -> (int)Math.round(Math.random() * 1000));
		new SimpleGraphVisualizationWindow<>(bf.getEventBus()).getPanel().setTooltipGenerator(n -> String.valueOf(n.getInternalLabel()));
		
		/* find solution */
		PerformanceLogger.logStart("search");
		List<TestNode> solutionPath = bf.nextSolution();
		PerformanceLogger.logEnd("search");
		assertNotNull(solutionPath);
		System.out.println("Generated " + bf.getCreatedCounter() + " nodes.");
		PerformanceLogger.printStatsAndClear(PerformanceMeasure.TIME);
		while (true);
	}

}
