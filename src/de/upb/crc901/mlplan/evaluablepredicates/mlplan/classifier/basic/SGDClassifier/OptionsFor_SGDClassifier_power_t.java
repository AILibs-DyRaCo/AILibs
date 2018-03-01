
    package de.upb.crc901.mlplan.evaluablepredicates.mlplan.classifier.basic.SGDClassifier;
    /*
        power_t : double
        The exponent for inverse scaling learning rate [default 0.5].


    */

    import de.upb.crc901.mlplan.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

    public class OptionsFor_SGDClassifier_power_t extends NumericRangeOptionPredicate {
        
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
    
