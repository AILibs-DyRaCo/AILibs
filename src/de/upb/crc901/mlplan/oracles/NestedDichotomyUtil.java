package de.upb.crc901.mlplan.oracles;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.upb.crc901.mlplan.core.MLPipeline;
import jaicore.basic.SetUtil;
import jaicore.ml.WekaUtil;
import jaicore.ml.classification.multiclass.reduction.splitters.RPNDSplitter;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class NestedDichotomyUtil {

	private static final Logger logger = LoggerFactory.getLogger(NestedDichotomyUtil.class);

	public static ClassSplit<String> createGeneralRPNDBasedSplit(Collection<String> classes, Random rand, String classifierName, Instances data) throws InterruptedException  {
		if (classes.size() < 2)
			throw new IllegalArgumentException("Cannot compute split for less than two classes!");
		RPNDSplitter splitter = new RPNDSplitter(data, rand);
		Collection<Collection<String>> splitAsCollection = null;
		try {
			splitAsCollection = splitter.split(classes, new MLPipeline(new Ranker(), new InfoGainAttributeEval(), AbstractClassifier.forName(classifierName, null)));
		} catch (InterruptedException e) {
			throw e;
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		Iterator<Collection<String>> it = splitAsCollection.iterator();
		return new ClassSplit<>(classes, it.next(), it.next());
	}

	public static ClassSplit<String> createGeneralRPNDBasedSplit(Collection<String> classes, Collection<String> s1, Collection<String> s2, Random rand, String classifierName,
			Instances data) {
		RPNDSplitter splitter = new RPNDSplitter(data, rand);
		Collection<Collection<String>> splitAsCollection = null;
		try {
			splitAsCollection = splitter.split(classes, s1, s2, AbstractClassifier.forName(classifierName, new String[] {}));
		} catch (Exception e) {
			e.printStackTrace();
		}
		Iterator<Collection<String>> it = splitAsCollection.iterator();
		return new ClassSplit<>(classes, it.next(), it.next());
	}

	public static ClassSplit<String> createUnaryRPNDBasedSplit(Collection<String> classes, Random rand, String classifierName, Instances data) {

		/* 2. if we have a leaf node, abort */
		if (classes.size() == 1)
			return new ClassSplit<>(classes, null, null);

		/* 3a. otherwise select randomly two classes */
		List<String> copy = new ArrayList<>(classes);
		Collections.shuffle(copy, rand);
		String c1 = copy.get(0);
		String c2 = copy.get(1);
		Collection<String> s1 = new HashSet<>();
		s1.add(c1);
		Collection<String> s2 = new HashSet<>();
		s2.add(c2);

		/* 3b. and 3c. train binary classifiers for c1 vs c2 */
		Instances reducedData = WekaUtil.mergeClassesOfInstances(data, s1, s2);
		Classifier c = null;
		try {
			c = AbstractClassifier.forName(classifierName, new String[] {});
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		try {
			c.buildClassifier(reducedData);
		} catch (Exception e) {
			e.printStackTrace();
		}

		/* 3d. insort the remaining classes */
		List<String> remainingClasses = new ArrayList<>(SetUtil.difference(SetUtil.difference(classes, s1), s2));
		int o1 = 0;
		int o2 = 0;
		for (int i = 0; i < remainingClasses.size(); i++) {
			String className = remainingClasses.get(i);
			Instances testData = WekaUtil.getInstancesOfClass(data, className);
			for (Instance inst : testData) {
				try {
					double prediction = c.classifyInstance(WekaUtil.getRefactoredInstance(inst));
					if (prediction == 0)
						o1++;
					else
						o2++;
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
		if (o1 > o2)
			s1.addAll(remainingClasses);
		else
			s2.addAll(remainingClasses);
		return new ClassSplit<>(classes, s1, s2);
	}
}
