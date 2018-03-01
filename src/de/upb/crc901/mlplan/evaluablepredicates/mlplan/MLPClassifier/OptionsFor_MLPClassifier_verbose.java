
package de.upb.crc901.mlplan.evaluablepredicates.mlplan.MLPClassifier;
/*
    verbose : bool, optional, default False
        Whether to print progress messages to stdout.


 */

import de.upb.crc901.mlplan.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

public class OptionsFor_MLPClassifier_verbose extends NumericRangeOptionPredicate {
	
	@Override
	protected double getMin() {
		return 0;
	}

	@Override
	protected double getMax() {
		return 0;
	}

	@Override
	protected int getSteps() {
		return 1;
	}

	@Override
	protected boolean needsIntegers() {
		return true;
	}
}

