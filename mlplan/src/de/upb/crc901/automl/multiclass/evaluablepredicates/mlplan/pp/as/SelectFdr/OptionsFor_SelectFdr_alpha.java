
    package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.pp.as.SelectFdr;
    /*
        alpha : float, optional
        The highest uncorrected p-value for features to keep.


    Attributes
    
    */

    import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

    public class OptionsFor_SelectFdr_alpha extends NumericRangeOptionPredicate {
        
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
    
