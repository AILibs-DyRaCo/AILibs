
    package de.upb.crc901.mlplan.evaluablepredicates.mlplan.pp.pca.MiniBatchSparsePCA;
    /*
        alpha : int,
        Sparsity controlling parameter. Higher values lead to sparser
        components.


    */

    import de.upb.crc901.mlplan.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

    public class OptionsFor_MiniBatchSparsePCA_alpha extends NumericRangeOptionPredicate {
        
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
    
