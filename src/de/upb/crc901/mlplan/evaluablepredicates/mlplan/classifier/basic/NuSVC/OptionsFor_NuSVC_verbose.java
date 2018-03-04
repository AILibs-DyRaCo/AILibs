package de.upb.crc901.mlplan.evaluablepredicates.mlplan.classifier.basic.NuSVC;

    import java.util.Arrays;
    import java.util.List;

    import de.upb.crc901.mlplan.evaluablepredicates.mlplan.OptionsPredicate;
    /*
        verbose : bool, default: False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.


    */
    public class OptionsFor_NuSVC_verbose extends OptionsPredicate {
        
        private static List<Object> validValues = Arrays.asList(new Object[]{}); // deactivate this option (always use default)

        @Override
        protected List<? extends Object> getValidValues() {
            return validValues;
        }
    }
    
