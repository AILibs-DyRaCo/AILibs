package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.classifier.basic.PassiveAggressiveClassifier;

    import java.util.Arrays;
    import java.util.List;

import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.OptionsPredicate;
    /*
        shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.


    */
    public class OptionsFor_PassiveAggressiveClassifier_shuffle extends OptionsPredicate {
        
        private static List<Object> validValues = Arrays.asList(new Object[]{"false"}); // true is default

        @Override
        protected List<? extends Object> getValidValues() {
            return validValues;
        }
    }
    
