package de.upb.crc901.mlplan.evaluablepredicates.mlplan.SGDClassifier;

import java.util.Arrays;
import java.util.List;

import de.upb.crc901.mlplan.evaluablepredicates.mlplan.OptionsPredicate;
/*
    learning_rate : string, optional
        The learning rate schedule:

        - 'constant': eta = eta0
        - 'optimal': eta = 1.0 / (alpha * (t + t0)) [default]
        - 'invscaling': eta = eta0 / pow(t, power_t)

        where t0 is chosen by a heuristic proposed by Leon Bottou.


 */
public class OptionsFor_SGDClassifier_learning_rate extends OptionsPredicate {
	
	private static List<Integer> validValues = Arrays.asList(new Integer[]{1, 2, 3});

	@Override
	protected List<? extends Object> getValidValues() {
		return validValues;
	}
}

