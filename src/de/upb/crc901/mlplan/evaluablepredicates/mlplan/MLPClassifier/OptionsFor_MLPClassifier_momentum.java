
package de.upb.crc901.mlplan.evaluablepredicates.mlplan.MLPClassifier;
/*
    momentum : float, default 0.9
        Momentum for gradient descent update. Should be between 0 and 1. Only
        used when solver='sgd'.


 */

import de.upb.crc901.mlplan.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

public class OptionsFor_MLPClassifier_momentum extends NumericRangeOptionPredicate {
	
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

