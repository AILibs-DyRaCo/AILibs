package jaicore.planning.graphgenerators.task.ceociptfd;

import java.util.Collection;
import java.util.List;
import java.util.Map;

import jaicore.logic.fol.structure.ConstantParam;
import jaicore.logic.fol.structure.Monom;
import jaicore.logic.fol.structure.VariableParam;

public interface EvaluablePredicate {
	
	public Collection<List<ConstantParam>> getParamsForPositiveEvaluation(Monom state, ConstantParam... partialGrounding);
	
	public boolean isOracable();
	
	public Collection<List<ConstantParam>> getParamsForNegativeEvaluation(Monom state, ConstantParam... partialGrounding);
	
	public boolean test(Monom state, ConstantParam... params); // usually we would evaluate ONLY the predicate with terms, but there may be terms that are described indirectly in the state.
}
