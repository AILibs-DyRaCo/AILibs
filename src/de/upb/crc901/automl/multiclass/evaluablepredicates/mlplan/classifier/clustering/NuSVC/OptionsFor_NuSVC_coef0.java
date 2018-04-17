
    package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.classifier.clustering.NuSVC;

    import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

    /*
        coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.


    */
    public class OptionsFor_NuSVC_coef0 extends NumericRangeOptionPredicate {
        
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
    
