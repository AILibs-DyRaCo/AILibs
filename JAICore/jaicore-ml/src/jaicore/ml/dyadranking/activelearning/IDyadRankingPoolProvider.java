package jaicore.ml.dyadranking.activelearning;

import java.util.Set;

import de.upb.isys.linearalgebra.Vector;
import jaicore.ml.activelearning.IActiveLearningPoolProvider;
import jaicore.ml.dyadranking.Dyad;

/**
 * Interface for an active learning pool provider in the context of dyad
 * ranking. It offers access to the pool of dyads both by instance features and
 * alternative features.
 * 
 * @author Jonas Hanselle
 *
 */
public interface IDyadRankingPoolProvider extends IActiveLearningPoolProvider {

	/**
	 * Returns the set of all {@link Dyad}s with the given {@link Vector} of
	 * instance features.
	 * 
	 * @param instanceFeatures {@link Vector} of instance features.
	 * @return {@link Set} of dyads with the given {@link Vector} of instance
	 *         features.
	 */
	public Set<Dyad> getDyadsByInstance(Vector instanceFeatures);

	/**
	 * Returns the set of all {@link Dyad}s with the given {@link Vector} of
	 * alternative features.
	 * 
	 * @param alternativeFeatures {@link Vector} of alternative features.
	 * @return {@link Set} of dyads with the given {@link Vector} of alternative
	 *         features.
	 */
	public Set<Dyad> getDyadsByAlternative(Vector alternativeFeatures);
}
