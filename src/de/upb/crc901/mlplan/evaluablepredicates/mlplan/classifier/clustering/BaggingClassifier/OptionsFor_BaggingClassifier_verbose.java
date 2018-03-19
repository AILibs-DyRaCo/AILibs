
    package de.upb.crc901.mlplan.evaluablepredicates.mlplan.classifier.clustering.BaggingClassifier;

    import de.upb.crc901.mlplan.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

    /*
        verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    
    */
    public class OptionsFor_BaggingClassifier_verbose extends NumericRangeOptionPredicate {
        
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
            return true; // already set by generator
        }

        @Override
        protected boolean isLinear() {
			return true;
		}
    }
    
