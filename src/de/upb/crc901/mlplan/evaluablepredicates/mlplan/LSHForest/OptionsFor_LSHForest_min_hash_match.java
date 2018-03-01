
package de.upb.crc901.mlplan.evaluablepredicates.mlplan.LSHForest;
/*
    min_hash_match : int (default = 4)
        lowest hash length to be searched when candidate selection is
        performed for nearest neighbors.


 */

import de.upb.crc901.mlplan.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

public class OptionsFor_LSHForest_min_hash_match extends NumericRangeOptionPredicate {
	
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
		return true;
	}
}

