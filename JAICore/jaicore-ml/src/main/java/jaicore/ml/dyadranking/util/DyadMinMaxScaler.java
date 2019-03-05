package jaicore.ml.dyadranking.util;

import java.util.List;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import jaicore.ml.core.dataset.IInstance;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;


/**
 * A scaler that can be fit to a certain dataset and then be used to normalize
 * dyad datasets, i.e. transform the data such that the values of each feature
 * lie between 0 and 1.
 * 
 * For feature x: x = x - x_min / (x_max - x_min)
 * 
 * @author Michael Braun, Mirko Jürgens
 *
 */

public class DyadMinMaxScaler extends AbstractDyadScaler {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1319262573945961139L;

	/**
	 * Transforms only the instances of each dyad according to the mean and standard
	 * of the data the scaler has been fit to.
	 * 
	 * @param dataset
	 *            The dataset of which the instances are to be standardized.
	 */
	@Override
	public void transformInstances(DyadRankingDataset dataset, List<Integer> ignoredIndices) {
		int lengthX = dataset.get(0).getDyadAtPosition(0).getInstance().length();
		for (IInstance instance : dataset) {
			IDyadRankingInstance drInstance = (IDyadRankingInstance) instance;
			for (Dyad dyad : drInstance) {
				for (int i = 0; i < lengthX; i++) {
					double value = dyad.getInstance().getValue(i);
					value -= statsX[i].getMin();
					value /= statsX[i].getMax() - statsX[i].getMin();
					dyad.getInstance().setValue(i, value);
				}
			}
		}
	}

	/**
	 * Transforms only the alternatives of each dyad according to the mean and
	 * standard deviation of the data the scaler has been fit to.
	 * 
	 * @param dataset
	 *            The dataset of which the alternatives are to be standardized.
	 */
	@Override
	public void transformAlternatives(DyadRankingDataset dataset, List<Integer> ignoredIndices) {
		int lengthY = dataset.get(0).getDyadAtPosition(0).getAlternative().length();
		for (IInstance instance : dataset) {
			IDyadRankingInstance drInstance = (IDyadRankingInstance) instance;
			for (Dyad dyad : drInstance) {
				for (int i = 0; i < lengthY; i++) {
					if (!ignoredIndices.contains(i)) {
						double value = dyad.getAlternative().getValue(i);
						value -= statsY[i].getMin();
						value /= statsY[i].getMax() - statsY[i].getMin();
						dyad.getAlternative().setValue(i, value);
					}
				}
			}
		}
	}

	public void untransform(DyadRankingDataset dataset) {
		int lengthX = dataset.get(0).getDyadAtPosition(0).getInstance().length();
		int lengthY = dataset.get(0).getDyadAtPosition(0).getAlternative().length();

		if (lengthX != statsX.length || lengthY != statsY.length)
			throw new IllegalArgumentException("The scaler was fit to dyads with instances of length " + statsX.length
					+ " and alternatives of length " + statsY.length + "\n but received instances of length " + lengthX
					+ " and alternatives of length " + lengthY);

		untransformInstances(dataset);
		untransformAlternatives(dataset);
	}

	/**
	 * Undoes the transformation of the instances of each dyad.
	 * 
	 * @param dataset
	 */
	public void untransformInstances(DyadRankingDataset dataset) {
		for (IInstance instance : dataset) {
			IDyadRankingInstance drInstance = (IDyadRankingInstance) instance;
			for (Dyad dyad : drInstance) {
				untransformInstance(dyad);
			}
		}
	}

	/**
	 * Undoes the transformation of the instance of a single dyad.
	 * 
	 * @param dyad
	 */
	public void untransformInstance(Dyad dyad) {
		int lengthX = dyad.getInstance().length();
		if (lengthX != statsX.length) {
			throw new IllegalArgumentException("The scaler was fit to instances of length " + statsX.length
					+ " but received an instance of length " + lengthX + ".");
		}
		for (int i = 0; i < lengthX; i++) {
			double value = dyad.getInstance().getValue(i);
			value *= statsX[i].getMax() - statsX[i].getMin();
			value += statsX[i].getMin();
			dyad.getInstance().setValue(i, value);
		}
	}

	/**
	 * Undoes the transformation of the alternatives of each dyad.
	 * 
	 * @param dataset
	 */
	public void untransformAlternatives(DyadRankingDataset dataset) {
		for (IInstance instance : dataset) {
			IDyadRankingInstance drInstance = (IDyadRankingInstance) instance;
			for (Dyad dyad : drInstance) {
				untransformAlternative(dyad);
			}
		}
	}

	/**
	 * Undoes the transformation on the alternative of a single dyad.
	 * 
	 * @param dyad
	 */
	public void untransformAlternative(Dyad dyad) {
		int lengthY = dyad.getAlternative().length();
		if (lengthY != statsY.length) {
			throw new IllegalArgumentException("The scaler was fit to alternatives of length " + statsY.length
					+ " but received an alternative of length " + lengthY + ".");
		}
		for (int i = 0; i < lengthY; i++) {
			double value = dyad.getAlternative().getValue(i);
			value *= statsY[i].getMax() - statsY[i].getMin();
			value += statsY[i].getMin();
			dyad.getAlternative().setValue(i, value);
		}
	}

	/**
	 * Prints the maxima of all features this scaler has been fit to.
	 */
	public void printMaxima() {
		if (statsX == null || statsY == null)
			throw new IllegalStateException("The scaler must be fit before calling this method!");
		System.out.print("Standard deviations for instances: ");
		for (SummaryStatistics stats : statsX) {
			System.out.print(stats.getMax() + ", ");
		}
		System.out.println();
		System.out.print("Standard deviations for alternatives: ");
		for (SummaryStatistics stats : statsY) {
			System.out.print(stats.getMax() + ", ");
		}
		System.out.println();
	}

	/**
	 * Prints the minima of all features this scaler has been fit to.
	 */
	public void printMinima() {
		if (statsX == null || statsY == null)
			throw new IllegalStateException("The scaler must be fit before calling this method!");
		System.out.print("Means for instances: ");
		for (SummaryStatistics stats : statsX) {
			System.out.print(stats.getMin() + ", ");
		}
		System.out.println();
		System.out.print("Means for alternatives: ");
		for (SummaryStatistics stats : statsY) {
			System.out.print(stats.getMin() + ", ");
		}
		System.out.println();
	}

}
