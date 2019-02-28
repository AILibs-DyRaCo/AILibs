package jaicore.ml.metafeatures;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import de.upb.isys.linearalgebra.Vector;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.trees.RandomTree.Tree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class RandomTreePerformanceBasedFeatureGenerator implements IPerformanceDecisionTreeBasedFeatureGenerator {

	private Tree tree;
	private Map<Tree, Integer> nodesIndices = new HashMap<>();
	private boolean allowUnsetValues = false;
	private double incomingUnsetValueValue = Double.NaN;
	private double outgoingUnsetValueValue = 0;
	private double occurenceValue = 1;
	private double nonOccurenceValue = -1;

	public void train(Map<Vector, Double> intermediatePipelineRepresentationsWithPerformanceValues) throws Exception {
		// Step 1: Transform to Instances Object
		ArrayList<Attribute> attInfo = new ArrayList<>();
		for (int i = 0; i < intermediatePipelineRepresentationsWithPerformanceValues.keySet().toArray(new Vector [0])[0].length() ; i++) {
			attInfo.add(new Attribute("Attribute-" + String.valueOf(i)));
		}
		Instances train = new Instances("train", attInfo, intermediatePipelineRepresentationsWithPerformanceValues.size());
		train.setClassIndex(train.numAttributes() - 1);
		intermediatePipelineRepresentationsWithPerformanceValues.forEach((features, value) -> {
			double [] values = new double [features.length() + 1];
			for (int i = 0; i < features.length(); i++) {
				values[i] = features.getValue(i);
			}
			values[values.length - 1] = value;
			train.add(new DenseInstance(1, values));
		});

		// Step 2: Train Random Tree
		RandomTree randomTree = new RandomTree();
		randomTree.buildClassifier(train);

		// Step 3: Count the nodes in the tree (DF Traversal Index Mapping)
		addIndexToMap(0, randomTree.getM_Tree());
	}

	private void addIndexToMap(int subTreeIndex, Tree subTree) {
		nodesIndices.put(subTree, subTreeIndex);

		if (subTree.getM_Successors() != null) {
			for (int i = 0; i < subTree.getM_Successors().length; i++) {
				addIndexToMap(++subTreeIndex, subTree.getM_Successors()[i]);
			}
		}
	}

	public Vector predict(Vector intermediatePipelineRepresentation) {
		Vector pipelineRepresentation = new DenseDoubleVector(nodesIndices.size(), nonOccurenceValue);

		// Query the RandomTree
		Tree subTree = tree;
		while (subTree != null) {	
			if (allowUnsetValues && !isValueUnset(intermediatePipelineRepresentation.getValue(subTree.getM_Attribute())) || !allowUnsetValues) {
				// The current node occurs
				pipelineRepresentation.setValue(nodesIndices.get(subTree), occurenceValue);			
				
				if (subTree.getM_Attribute() == -1) {
					// We are at a leaf - stop
					subTree = null;
				} else {
					if (intermediatePipelineRepresentation.getValue(subTree.getM_Attribute()) < subTree
							.getM_SplitPoint()) {
						// we go to the left
						subTree = subTree.getM_Successors()[0];
					} else {
						// we go to the right
						subTree = subTree.getM_Successors()[1];
					}
				}
				
			} else {
				// We do allow unset values and the value is unset - set the subtree to non occurence and end the traversal
				setSubTreeToValue(subTree, outgoingUnsetValueValue, pipelineRepresentation);
				subTree = null;
			}
		}

		return pipelineRepresentation;
	}

	private boolean isValueUnset(double value) {
		if (Double.isNaN(incomingUnsetValueValue)) {
			return Double.isNaN(value);
		} else {
			return value == incomingUnsetValueValue;
		}
	}

	private void setSubTreeToValue(Tree subTree, double value, Vector featureRepresentation) {
		featureRepresentation.setValue(nodesIndices.get(subTree), value);

		if (subTree.getM_Successors() != null) {
			for (int i = 0; i < subTree.getM_Successors().length; i++) {
				setSubTreeToValue(subTree.getM_Successors()[i], value, featureRepresentation);
			}
		}
	}
}
