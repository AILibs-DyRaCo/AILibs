package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.classifier.clustering.MLPClassifier;

    import java.util.Arrays;
    import java.util.List;

import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.OptionsPredicate;
    /*
        nesterovs_momentum : boolean, default True
        Whether to use Nesterov's momentum. Only used when solver='sgd' and
        momentum > 0.


    */
    public class OptionsFor_MLPClassifier_nesterovs_momentum extends OptionsPredicate {
        
        private static List<Object> validValues = Arrays.asList(new Object[]{});

        @Override
        protected List<? extends Object> getValidValues() {
            return validValues;
        }
    }
    
