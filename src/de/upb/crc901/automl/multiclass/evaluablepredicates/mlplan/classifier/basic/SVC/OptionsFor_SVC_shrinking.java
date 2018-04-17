package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.classifier.basic.SVC;

    import java.util.Arrays;
    import java.util.List;

import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.OptionsPredicate;
    /*
        shrinking : boolean, optional (default=True)
        Whether to use the shrinking heuristic.


    */
    public class OptionsFor_SVC_shrinking extends OptionsPredicate {
        
        private static List<Object> validValues = Arrays.asList(new Object[]{ "false"}); // true is default

        @Override
        protected List<? extends Object> getValidValues() {
            return validValues;
        }
    }
    
