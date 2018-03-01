
package de.upb.crc901.mlplan.evaluablepredicates.mlplan.BaggingClassifier;
/*
    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    
 */

import de.upb.crc901.mlplan.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

public class OptionsFor_BaggingClassifier_verbose extends NumericRangeOptionPredicate {
	
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

