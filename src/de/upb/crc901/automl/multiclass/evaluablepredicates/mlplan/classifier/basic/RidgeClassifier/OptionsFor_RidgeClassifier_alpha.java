
    package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.classifier.basic.RidgeClassifier;
    /*
        alpha : float
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Alpha corresponds to ``C^-1`` in other linear models such as
        LogisticRegression or LinearSVC.


    */

    import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

    public class OptionsFor_RidgeClassifier_alpha extends NumericRangeOptionPredicate {
        
        @Override
        protected double getMin() {
            return 1e-3;
        }

        @Override
        protected double getMax() {
            return 1e3;
        }

        @Override
        protected int getSteps() {
            return 7;
        }

        @Override
        protected boolean needsIntegers() {
            return false;
        }
    }
    
