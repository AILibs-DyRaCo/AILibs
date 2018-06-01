package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.pp.as.Isomap;

    import java.util.Arrays;
    import java.util.List;

import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.OptionsPredicate;
    /*
        eigen_solver : ["auto"|"arpack"|"dense"]
        "auto" : Attempt to choose the most efficient solver
        for the given problem.

        "arpack" : Use Arnoldi decomposition to find the eigenvalues
        and eigenvectors.

        "dense" : Use a direct solver (i.e. LAPACK)
        for the eigenvalue decomposition.


    */
    public class OptionsFor_Isomap_eigen_solver extends OptionsPredicate {
        
        private static List<Object> validValues = Arrays.asList(new Object[]{"auto","arpack","dense"});

        @Override
        protected List<? extends Object> getValidValues() {
            return validValues;
        }
    }
    
