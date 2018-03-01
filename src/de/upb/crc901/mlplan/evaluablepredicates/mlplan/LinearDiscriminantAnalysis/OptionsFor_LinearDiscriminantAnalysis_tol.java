
package de.upb.crc901.mlplan.evaluablepredicates.mlplan.LinearDiscriminantAnalysis;
/*
    tol : float, optional, (default 1.0e-4)
        Threshold used for rank estimation in SVD solver.

        .. versionadded:: 0.17

    Attributes
    
 */

import de.upb.crc901.mlplan.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

public class OptionsFor_LinearDiscriminantAnalysis_tol extends NumericRangeOptionPredicate {
	
	@Override
	protected double getMin() {
		return 1;
	}

	@Override
	protected double getMax() {
		return 1;
	}

	@Override
	protected int getSteps() {
		return -1;
	}

	@Override
	protected boolean needsIntegers() {
		return false;
	}
}

