package jaicore.search.algorithms.standard.core;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jaicore.search.algorithms.interfaces.IObservableORGraphSearch;
import jaicore.search.algorithms.interfaces.IObservableORGraphSearchFactory;
import jaicore.search.algorithms.interfaces.IPathUnification;
import jaicore.search.algorithms.interfaces.ISolutionEvaluator;
import jaicore.search.algorithms.standard.uncertainty.BasicUncertaintySource;
import jaicore.search.algorithms.standard.uncertainty.UncertaintyRandomCompletionEvaluator;
import jaicore.search.algorithms.standard.uncertainty.explorationexploitationsearch.BasicExplorationCandidateSelector;
import jaicore.search.algorithms.standard.uncertainty.explorationexploitationsearch.UncertaintyExplorationOpenSelection;
import jaicore.search.algorithms.standard.uncertainty.explorationexploitationsearch.IPhaseLengthAdjuster.AdjustmentOptions;
import jaicore.search.algorithms.standard.uncertainty.paretosearch.ParetoSelection;
import jaicore.search.structure.core.GraphGenerator;
import jaicore.search.structure.core.Node;
import jaicore.logic.fol.structure.ConstantParam;
import jaicore.logic.fol.structure.VariableParam;
import jaicore.planning.graphgenerators.task.tfd.TFDNode;

public class UncertaintyORGraphSearchFactory <T, A> implements IObservableORGraphSearchFactory<T, A, Double>{

	private static final Logger logger = LoggerFactory.getLogger(UncertaintyORGraphSearchFactory.class);
	
	public enum OversearchAvoidanceMode {
		PARETO_FRONT_SELECTION,
		TWO_PHASE_SELECTION,
		NONE
	}

	private OversearchAvoidanceMode oversearchAvoidanceMode;
	private IPathUnification<T> pathUnification;
	private int timeoutForFInMS;
	private INodeEvaluator<T, Double> timeoutEvaluator;
	private String loggerName;
	
	public UncertaintyORGraphSearchFactory(OversearchAvoidanceMode oversearchAvoidanceMode, IPathUnification<T> pathUnification) {
		this.oversearchAvoidanceMode = oversearchAvoidanceMode;
		this.pathUnification = pathUnification;
	}
	
	@Override
	public IObservableORGraphSearch<T, A, Double> createSearch(GraphGenerator<T, A> graphGenerator, INodeEvaluator<T, Double> nodeEvaluator, int numberOfCPUs) {
		ORGraphSearch<T, A, Double> search;
		if (oversearchAvoidanceMode == OversearchAvoidanceMode.NONE) {
			search = new ORGraphSearch<>(graphGenerator, nodeEvaluator);
		} else {
			search = new ORGraphSearch<>(
					graphGenerator,
					new UncertaintyRandomCompletionEvaluator<T, A, Double>(
						new Random(123l),
						3,
						pathUnification,
						new ISolutionEvaluator<T, Double>() {
							@Override
							public Double evaluateSolution(List<T> solutionPath) {
								// TODO: Check if this works
								try {
									return nodeEvaluator.f(new Node<T, Double>(null, solutionPath.get(solutionPath.size() - 1)));
								} catch (Throwable e) {
									logger.error(e.getMessage());
									return 1.0d;
								}
							}
			
							@Override
							public boolean doesLastActionAffectScoreOfAnySubsequentSolution(
									List<T> partialSolutionPath) {
								return true;
							}
						},
						new BasicUncertaintySource<T>()
					)
			);
			
			if (oversearchAvoidanceMode == OversearchAvoidanceMode.TWO_PHASE_SELECTION) {
				search.setOpen(new UncertaintyExplorationOpenSelection<T, Double>(
						10,
						20,
						0.05d,
						0.05d,
						(currentExplorationLength, currentExploitationLength, lastExplorationPhaseCompletelyUsed, phaseSwitchAmount) -> {
							if (phaseSwitchAmount < 10 && lastExplorationPhaseCompletelyUsed) {
								return AdjustmentOptions.INCREASE_EXPLORATION_PHASE_LENGTH;
							} else {
								if (lastExplorationPhaseCompletelyUsed && (currentExploitationLength - currentExplorationLength) > 10) {
									return AdjustmentOptions.INCREASE_EXPLORATION_PHASE_LENGTH;
								} else {
									return AdjustmentOptions.INCREASE_EXPLOITATION_PHASE_LENGTH;
								}
							}
						},
						(solution1, solution2) -> {
							if (solution1 instanceof TFDNode && solution2 instanceof TFDNode) {
//								TFDNode t1 = (TFDNode) solution1;
//								TFDNode t2 = (TFDNode) solution2;
//								Set<Entry<VariableParam, ConstantParam>> s1 = t1.getAppliedMethodInstance().getGrounding().entrySet();
//								List<String> l1 = new ArrayList<>();
//								s1.forEach(e -> l1.add(e.getValue().getName()));
//								Set<Entry<VariableParam, ConstantParam>> s2 = t2.getAppliedMethodInstance().getGrounding().entrySet();
//								List<String> l2 = new ArrayList<>();
//								s2.forEach(e -> l2.add(e.getValue().getName()));
//								List<String> u = new ArrayList<>();
//								u.addAll(l1);
//								u.addAll(l2);
//								Set<String> i = new HashSet<>();
//								l1.forEach(e -> {
//									if (l2.contains(e)) {
//										i.add(e);
//									}
//								});
//								return 1.0d - (((double)(u.size() - i.size())) / ((double) u.size()));
								return 0.0d;
							} else {
								return 0.0d;
							}
						},
						new BasicExplorationCandidateSelector<T, Double>(5.0d)
				));
			} else {
				new ParetoSelection<>(false);
			}
		}
		
		search.parallelizeNodeExpansion(numberOfCPUs);
		search.setTimeoutForComputationOfF(this.timeoutForFInMS, this.timeoutEvaluator);
		if (loggerName != null && loggerName.length() > 0)
			search.setLoggerName(loggerName);
		return search;
	}
	
	public void setTimeoutForFComputation(final int timeoutInMS, final INodeEvaluator<T, Double> timeoutEvaluator) {
		this.timeoutForFInMS = timeoutInMS;
		this.timeoutEvaluator = timeoutEvaluator;
	}

	public int getTimeoutForFInMS() {
		return this.timeoutForFInMS;
	}

	public INodeEvaluator<T, Double> getTimeoutEvaluator() {
		return this.timeoutEvaluator;
	}

	public String getLoggerName() {
		return loggerName;
	}

	public void setLoggerName(String loggerName) {
		this.loggerName = loggerName;
	}
	
}
