
    package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.pp.as.LocallyLinearEmbedding;
    /*
        n_components : integer
        number of coordinates for the manifold


    */

    import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

    public class OptionsFor_LocallyLinearEmbedding_n_components extends NumericRangeOptionPredicate {
        
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
    
