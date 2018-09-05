package hasco.core;

import hasco.optimizingfactory.SoftwareConfigurationAlgorithmFactory;
import jaicore.basic.algorithm.AlgorithmProblemTransformer;
import jaicore.planning.graphgenerators.IPlanningGraphGeneratorDeriver;
import jaicore.planning.model.ceoc.CEOCAction;
import jaicore.planning.model.ceoc.CEOCOperation;
import jaicore.planning.model.task.ceocipstn.CEOCIPSTNPlanningProblem;
import jaicore.planning.model.task.ceocstn.OCMethod;
import jaicore.search.core.interfaces.IGraphSearchFactory;
import jaicore.search.model.probleminputs.GraphSearchProblemInput;

public class HASCOFactory<ISearch, N,A,V extends Comparable<V>> implements SoftwareConfigurationAlgorithmFactory<RefinementConfiguredSoftwareConfigurationProblem<V>, HASCORunReport<V>, V> {

	private RefinementConfiguredSoftwareConfigurationProblem<V> problem;
	private IPlanningGraphGeneratorDeriver<CEOCOperation, OCMethod, CEOCAction, CEOCIPSTNPlanningProblem<CEOCOperation, OCMethod, CEOCAction>, N, A> planningGraphGeneratorDeriver;
	private IGraphSearchFactory<ISearch, ?, N, A, V, ?, ?> searchFactory;
	private AlgorithmProblemTransformer<GraphSearchProblemInput<N, A, V>, ISearch> searchProblemTransformer;
	
	@Override
	public <P> void setProblemInput(P problemInput, AlgorithmProblemTransformer<P, RefinementConfiguredSoftwareConfigurationProblem<V>> reducer) {
		setProblemInput(reducer.transform(problemInput));
	}

	@Override
	public void setProblemInput(RefinementConfiguredSoftwareConfigurationProblem<V> problemInput) {
		this.problem = problemInput;
	}

	@Override
	public HASCO<ISearch, N, A, V> getAlgorithm() {
		return new HASCO<>(problem, planningGraphGeneratorDeriver, searchFactory, searchProblemTransformer);
	}

	public IPlanningGraphGeneratorDeriver<CEOCOperation, OCMethod, CEOCAction, CEOCIPSTNPlanningProblem<CEOCOperation, OCMethod, CEOCAction>, N, A> getPlanningGraphGeneratorDeriver() {
		return planningGraphGeneratorDeriver;
	}

	public void setPlanningGraphGeneratorDeriver(
			IPlanningGraphGeneratorDeriver<CEOCOperation, OCMethod, CEOCAction, CEOCIPSTNPlanningProblem<CEOCOperation, OCMethod, CEOCAction>, N, A> planningGraphGeneratorDeriver) {
		this.planningGraphGeneratorDeriver = planningGraphGeneratorDeriver;
	}

	public IGraphSearchFactory<ISearch, ?, N, A, V, ?, ?> getSearchFactory() {
		return searchFactory;
	}

	public void setSearchFactory(IGraphSearchFactory<ISearch, ?, N, A, V, ?, ?> searchFactory) {
		this.searchFactory = searchFactory;
	}

	public AlgorithmProblemTransformer<GraphSearchProblemInput<N, A, V>, ISearch> getSearchProblemTransformer() {
		return searchProblemTransformer;
	}

	public void setSearchProblemTransformer(AlgorithmProblemTransformer<GraphSearchProblemInput<N, A, V>, ISearch> searchProblemTransformer) {
		this.searchProblemTransformer = searchProblemTransformer;
	}	
}
