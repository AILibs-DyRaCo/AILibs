
package de.upb.crc901.mlplan.evaluablepredicates.mlplan.SVC;
/*
    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.


 */

import de.upb.crc901.mlplan.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

public class OptionsFor_SVC_tol extends NumericRangeOptionPredicate {
	
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
		return false;
	}
}

