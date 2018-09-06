package hasco.core;

import java.util.List;

import jaicore.planning.graphgenerators.IPlanningGraphGeneratorDeriver;
import jaicore.planning.model.ceoc.CEOCAction;
import jaicore.planning.model.ceoc.CEOCOperation;
import jaicore.planning.model.core.Plan;
import jaicore.planning.model.task.ceocipstn.CEOCIPSTNPlanningProblem;
import jaicore.planning.model.task.ceocstn.OCMethod;
import jaicore.search.core.interfaces.GraphGenerator;

/**
 * This class only serves to facilitate the usage of HASCO when passing a IPlanningGraphGeneratorDeriver.
 * HASCO requires a IHASCOPlanningGraphGeneratorDeriver, which only takes away some of the generics of IPlanningGraphGeneratorDeriver,
 * but this implies that you cannot just use arbitrary IPlanningGraphGeneratorDeriver objects anymore.
 * To circumvent this problem, this class implements the IHASCOPlanningGraphGeneratorDeriver and wraps any IPlanningGraphGeneratorDeriver.
 * 
 * @author fmohr
 *
 * @param <N>
 * @param <A>
 */
public class DefaultHASCOPlanningGraphGeneratorDeriver<N,A> implements IHASCOPlanningGraphGeneratorDeriver<N, A> {

	private final IPlanningGraphGeneratorDeriver<CEOCOperation, OCMethod, CEOCAction, CEOCIPSTNPlanningProblem<CEOCOperation, OCMethod, CEOCAction>, N, A> wrappedDeriver;
	
	public DefaultHASCOPlanningGraphGeneratorDeriver(
			IPlanningGraphGeneratorDeriver<CEOCOperation, OCMethod, CEOCAction, CEOCIPSTNPlanningProblem<CEOCOperation, OCMethod, CEOCAction>, N, A> wrappedDeriver) {
		super();
		this.wrappedDeriver = wrappedDeriver;
	}

	@Override
	public GraphGenerator<N, A> transform(CEOCIPSTNPlanningProblem<CEOCOperation, OCMethod, CEOCAction> problem) {
		return wrappedDeriver.transform(problem);
	}

	@Override
	public Plan<CEOCAction> getPlan(List<N> path) {
		return wrappedDeriver.getPlan(path);
	}

}
