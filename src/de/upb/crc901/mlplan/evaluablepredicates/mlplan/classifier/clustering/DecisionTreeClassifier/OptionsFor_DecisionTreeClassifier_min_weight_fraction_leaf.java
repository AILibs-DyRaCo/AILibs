
    package de.upb.crc901.mlplan.evaluablepredicates.mlplan.classifier.clustering.DecisionTreeClassifier;

    import de.upb.crc901.mlplan.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

    /*
        min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.


    */
    public class OptionsFor_DecisionTreeClassifier_min_weight_fraction_leaf extends NumericRangeOptionPredicate {
        
        @Override
        protected double getMin() {
            return 1
                ;
        }

        @Override
        protected double getMax() {
            return 1
                ;
        }

        @Override
        protected int getSteps() {
            return -1
                ;
        }

        @Override
        protected boolean needsIntegers() {
            return false; // already set by generator
        }

        @Override
        protected boolean isLinear() {
			return true;
		}
    }
    
