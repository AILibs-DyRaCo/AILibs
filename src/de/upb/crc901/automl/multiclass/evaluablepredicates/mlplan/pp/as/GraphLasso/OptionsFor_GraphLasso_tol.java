
    package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.pp.as.GraphLasso;
    /*
        tol : positive float, default 1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped.


    */

    import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

    public class OptionsFor_GraphLasso_tol extends NumericRangeOptionPredicate {
        
        @Override
        protected double getMin() {
            return 1e-5;
        }

        @Override
        protected double getMax() {
            return 1e-1;
        }

        @Override
        protected int getSteps() {
            return 5;
        }

        @Override
        protected boolean needsIntegers() {
            return false;
        }
    }
    
