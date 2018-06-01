
    package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.pp.as.LatentDirichletAllocation;
    /*
        verbose : int, optional (default=0)
        Verbosity level.


    */

    import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

    public class OptionsFor_LatentDirichletAllocation_verbose extends NumericRangeOptionPredicate {
        
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
    
