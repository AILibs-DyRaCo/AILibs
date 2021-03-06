package jaicore.ml.dyadranking.util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import de.upb.isys.linearalgebra.Vector;
import jaicore.ml.core.dataset.IInstance;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.DyadRankingInstance;
import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import jaicore.ml.dyadranking.dataset.SparseDyadRankingInstance;

/**
 * A scaler that can be fit to a certain dataset and then be used to standardize
 * datasets, i.e. transform the data to have a mean of 0 and a standard
 * deviation of 1 according to the data it was fit to.
 * 
 * @author Michael Braun, Jonas Hanselle, Mirko Jürgens, Helena Graf
 *
 */
public abstract class AbstractDyadScaler implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -825893010030419116L;

	protected SummaryStatistics[] statsX;
	protected SummaryStatistics[] statsY;

	public SummaryStatistics[] getStatsX() {
		return statsX;
	}

	public SummaryStatistics[] getStatsY() {
		return statsY;
	}

	/**
	 * Fits the standard scaler to the dataset.
	 * 
	 * @param dataset The dataset the scaler should be fit to.
	 */
	public void fit(DyadRankingDataset dataset) {
		int lengthX = dataset.get(0).getDyadAtPosition(0).getInstance().length();
		int lengthY = dataset.get(0).getDyadAtPosition(0).getAlternative().length();
		statsX = new SummaryStatistics[lengthX];
		statsY = new SummaryStatistics[lengthY];
		for (int i = 0; i < lengthX; i++) {
			statsX[i] = new SummaryStatistics();
		}
		for (int i = 0; i < lengthY; i++) {
			statsY[i] = new SummaryStatistics();
		}
		for (IInstance instance : dataset) {
			IDyadRankingInstance drInstance = (IDyadRankingInstance) instance;
			for (Dyad dyad : drInstance) {
				for (int i = 0; i < lengthX; i++) {
					statsX[i].addValue(dyad.getInstance().getValue(i));
				}
				for (int i = 0; i < lengthY; i++) {
					statsY[i].addValue(dyad.getAlternative().getValue(i));
				}
			}
		}
	}

	/**
	 * Transforms the entire dataset according to the mean and standard deviation of
	 * the data the scaler has been fit to.
	 * 
	 * @param dataset The dataset to be standardized.
	 */
	public void transform(DyadRankingDataset dataset) {
		int lengthX = dataset.get(0).getDyadAtPosition(0).getInstance().length();
		int lengthY = dataset.get(0).getDyadAtPosition(0).getAlternative().length();

		if (lengthX != statsX.length || lengthY != statsY.length)
			throw new IllegalArgumentException("The scaler was fit to dyads with instances of length " + statsX.length
					+ " and alternatives of length " + statsY.length + "\n but received instances of length " + lengthX
					+ " and alternatives of length " + lengthY);

		transformInstances(dataset);
		transformAlternatives(dataset);
	}

	/**
	 * Transforms only the instances of each dyad according to the mean and standard
	 * of the data the scaler has been fit to.
	 * 
	 * @param dataset The dataset of which the instances are to be standardized.
	 */
	public void transformInstances(DyadRankingDataset dataset) {
		transformInstances(dataset, new ArrayList<>());
	}

	/**
	 * Transforms only the alternatives of each dyad according to the mean and
	 * standard deviation of the data the scaler has been fit to.
	 * 
	 * @param dataset The dataset of which the alternatives are to be standardized.
	 */
	public void transformAlternatives(DyadRankingDataset dataset) {
		transformAlternatives(dataset, new ArrayList<>());
	}

	/**
	 * Transforms only the instances of each dyad according to the mean and standard
	 * deviation of the data the scaler has been fit to. The attributes with indices
	 * contained in ignoredIndices are not transformed. {
	 * 
	 * @param dataset        The dataset of which the alternatives are to be
	 *                       standardized.
	 * @param ignoredIndices The {@link List} of indices that are been ignored by
	 *                       the scaler.
	 */
	public abstract void transformInstances(Dyad dyad, List<Integer> ignoredIndices);

	/**
	 * Transforms only the alternatives of each dyad according to the mean and
	 * standard deviation of the data the scaler has been fit to.
	 * 
	 * @param dataset        The dataset of which the alternatives are to be
	 *                       standardized.
	 * @param ignoredIndices The {@link List} of indices that are been ignored by
	 *                       the scaler.
	 */
	public abstract void transformAlternatives(Dyad dyad, List<Integer> ignoredIndices);

	/**
	 * Transforms an instance feature vector.
	 * 
	 * @param Instance       vector to be transformed
	 * @param ignoredIndices
	 */
	public abstract void transformInstaceVector(Vector vector, List<Integer> ignoredIndices);

	/**
	 * Transforms only the instances of each dyad in a
	 * {@link SparseDyadRankingInstance} according to the mean and standard
	 * deviation of the data the scaler has been fit to. The attributes with indices
	 * contained in ignoredIndices are not transformed. {
	 * 
	 * @param dataset        The dataset of which the alternatives are to be
	 *                       standardized.
	 * @param ignoredIndices The {@link List} of indices that are been ignored by
	 *                       the scaler.
	 */
	public void transformInstances(SparseDyadRankingInstance drInstance, List<Integer> ignoredIndices) {
		transformInstaceVector(drInstance.getDyadAtPosition(0).getInstance(), ignoredIndices);
	}

	/**
	 * Transforms only the instances of each dyad in a
	 * {@link DyadRankingInstance} according to the mean and standard
	 * deviation of the data the scaler has been fit to. The attributes with indices
	 * contained in ignoredIndices are not transformed. {
	 * 
	 * @param dataset        The dataset of which the alternatives are to be
	 *                       standardized.
	 * @param ignoredIndices The {@link List} of indices that are been ignored by
	 *                       the scaler.
	 */
	public void transformInstances(DyadRankingInstance drInstance, List<Integer> ignoredIndices) {
		for (Dyad dyad : drInstance)
			transformInstances(dyad, ignoredIndices);
	}

	/**
	 * Transforms only the alternatives of each dyad in an
	 * {@link IDyadRankingInstance} according to the mean and standard
	 * deviation of the data the scaler has been fit to. The attributes with indices
	 * contained in ignoredIndices are not transformed. {
	 * 
	 * @param dataset        The dataset of which the alternatives are to be
	 *                       standardized.
	 * @param ignoredIndices The {@link List} of indices that are been ignored by
	 *                       the scaler.
	 */
	public void transformAlternatives(IDyadRankingInstance drInstance, List<Integer> ignoredIndices) {
		for (Dyad dyad : drInstance)
			transformAlternatives(dyad, ignoredIndices);
	}

	/**
	 * Transforms only the instances of each dyad in a
	 * {@link DyadRankingDataset} according to the mean and standard
	 * deviation of the data the scaler has been fit to. The attributes with indices
	 * contained in ignoredIndices are not transformed. {
	 * 
	 * @param dataset        The dataset of which the alternatives are to be
	 *                       standardized.
	 * @param ignoredIndices The {@link List} of indices that are been ignored by
	 *                       the scaler.
	 */
	public void transformInstances(DyadRankingDataset dataset, List<Integer> ignoredIndices) {
		for (IInstance instance : dataset) {
			if (instance instanceof SparseDyadRankingInstance) {
				SparseDyadRankingInstance drSparseInstance = (SparseDyadRankingInstance) instance;
				this.transformInstances(drSparseInstance, ignoredIndices);
			} else if (instance instanceof DyadRankingInstance) {
				DyadRankingInstance drDenseInstance = (DyadRankingInstance) instance;
				this.transformInstances(drDenseInstance, ignoredIndices);
			} else {
				throw new IllegalArgumentException(
						"The scalers only support SparseDyadRankingInstance and DyadRankingInstance!");
			}
		}
	}

	/**
	 * Transforms only the alternatives of each dyad in a
	 * {@link DyadRankingDataset} according to the mean and standard
	 * deviation of the data the scaler has been fit to. The attributes with indices
	 * contained in ignoredIndices are not transformed. {
	 * 
	 * @param dataset        The dataset of which the alternatives are to be
	 *                       standardized.
	 * @param ignoredIndices The {@link List} of indices that are been ignored by
	 *                       the scaler.
	 */
	public void transformAlternatives(DyadRankingDataset dataset, List<Integer> ignoredIndices) {
		for (IInstance instance : dataset) {
			IDyadRankingInstance drInstance = (IDyadRankingInstance) instance;
			this.transformAlternatives(drInstance, ignoredIndices);
		}
	}

	/**
	 * Fits the standard scaler to the dataset and transforms the entire dataset
	 * according to the mean and standard deviation of the dataset.
	 * 
	 * @param dataset The dataset to be standardized.
	 */

	public void fitTransform(DyadRankingDataset dataset) {
		this.fit(dataset);
		this.transform(dataset);
	}

	/**
	 * Prints the standard devations of all features this scaler has been fit to.
	 */
	public String getPrettySTDString() {
		if (statsX == null || statsY == null) {
			throw new IllegalStateException("The scaler must be fit before calling this method!");
		}
			
		StringBuilder builder = new StringBuilder();
		
		builder.append("Standard deviations for instances: ");
		for (SummaryStatistics stats : statsX) {
			builder.append(stats.getStandardDeviation());
			builder.append(", ");
		}
		builder.append(System.lineSeparator());
		
		builder.append("Standard deviations for alternatives: ");
		for (SummaryStatistics stats : statsY) {
			builder.append(stats.getStandardDeviation());
			builder.append(", ");
		}
		builder.append(System.lineSeparator());
		
		return builder.toString();
	}

	/**
	 * Returns a String for the means of all features this scaler has been fit to.
	 */
	public String getPrettyMeansString() {
		if (statsX == null || statsY == null) {
			throw new IllegalStateException("The scaler must be fit before calling this method!");
		}
		
		StringBuilder builder = new StringBuilder();
			
		builder.append("Means for instances: ");
		for (SummaryStatistics stats : statsX) {
			builder.append(stats.getMean());
			builder.append(", ");
		}
		builder.append(System.lineSeparator());
		
		builder.append("Means for alternatives: ");
		for (SummaryStatistics stats : statsY) {
			builder.append(stats.getMean());
			builder.append(", ");
		}
		builder.append(System.lineSeparator());
		
		return builder.toString();
	}
}
