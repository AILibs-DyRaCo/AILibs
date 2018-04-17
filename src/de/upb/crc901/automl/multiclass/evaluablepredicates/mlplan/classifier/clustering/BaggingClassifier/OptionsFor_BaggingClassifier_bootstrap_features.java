package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.classifier.clustering.BaggingClassifier;

    import java.util.Arrays;
    import java.util.List;

import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.OptionsPredicate;
    /*
        bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.


    */
    public class OptionsFor_BaggingClassifier_bootstrap_features extends OptionsPredicate {
        
        private static List<Object> validValues = Arrays.asList(new Object[]{});

        @Override
        protected List<? extends Object> getValidValues() {
            return validValues;
        }
    }
    
