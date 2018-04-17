
    package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.pp.as.SelectFwe;
    /*
        score_func : callable
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues).
        Default is f_classif (see below "See also"). The default function only
        works with classification tasks.


    */

    import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

    public class OptionsFor_SelectFwe_score_func extends NumericRangeOptionPredicate {
        
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
    
