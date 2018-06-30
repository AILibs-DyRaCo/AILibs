package jaicore.ml.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class FeatureSpace {

	private List<FeatureDomain> featureDomains;

	public FeatureSpace() {
		featureDomains = new ArrayList<>();
	}

	public FeatureSpace(Instances data) {
		this();
		for (int i = 0; i < data.numAttributes(); i++) {
			Attribute attr = data.attribute(i);
			if (attr.type() == Attribute.NUMERIC) {
				NumericFeatureDomain domain = new NumericFeatureDomain(false, -5000.0d, 5000.d);
				featureDomains.add(domain);
				System.out.println();
				// } else if(attr.type() == Attribute.NOMINAL) {
				// String[] values = attr.
				// CategoricalFeatureDomain domain = new CategoricalFeatureDomain()
			} else {
				// TODO add categorical features!!!
				throw new IllegalArgumentException("Attribute type not supported!");
			}

		}
	}

	public FeatureSpace(List<FeatureDomain> domains) {
		featureDomains = new ArrayList<FeatureDomain>();
		for (FeatureDomain domain : domains)
			if (domain instanceof NumericFeatureDomain) {
				NumericFeatureDomain numDomain = (NumericFeatureDomain) domain;
				FeatureDomain nDomain = new NumericFeatureDomain(numDomain.isInteger(), numDomain.getMin(), numDomain.getMax());
				featureDomains.add(nDomain);
			}
		// TODO add support for categorical features
	}

	public FeatureSpace(FeatureSpace space) {
		this(Arrays.asList(space.getFeatureDomains()));
	}

	public FeatureSpace(FeatureDomain[] domains) {
		this(Arrays.asList(domains));
	}

	public FeatureDomain[] toArray() {
		return (FeatureDomain[]) featureDomains.toArray();
	}

	public void add(FeatureDomain domain) {
		featureDomains.add(domain);
	}

	public FeatureDomain[] getFeatureDomains() {
		return featureDomains.toArray(new FeatureDomain[featureDomains.size()]);
	}

	public double getRangeSize() {
		double size = 1.0d;
		for (FeatureDomain domain : featureDomains)
			size *= domain.getRangeSize();
		return size;
	}

	public double getRangeSizeOfFeatureSubspace(int[] featureIndices) {
		double size = 1.0d;
		for (int featureIndex : featureIndices)
			size *= featureDomains.get(featureIndex).getRangeSize();
		return size;
	}

	public int getDimensionality() {
		return featureDomains.size();
	}

	public FeatureDomain getFeatureDomain(int index) {
		return featureDomains.get(index);
	}

	public boolean containsInstance(Instance instance) {
		boolean val = true;
		for (int i = 0; i < featureDomains.size(); i++) {
			FeatureDomain domain = featureDomains.get(i);
			val &= domain.contains(instance.value(i));
		}
		return val;
	}

}
