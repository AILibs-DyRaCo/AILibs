package de.upb.crc901.mlplan.multiclass.wekamlplan.weka;

import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.commons.math3.util.Pair;
import org.junit.Test;

import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import hasco.model.Component;
import hasco.serialization.ComponentLoader;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;

public class WekaMLPlanWekaUtil {

	@Test
	public static List<MLPipeline> getAllLegalWekaPipelinesWithDefaultConfig() {
		ArrayList<MLPipeline> result = new ArrayList<MLPipeline>();
//			[weka.attributeSelection.Ranker, weka.attributeSelection.CorrelationAttributeEval]
//					[weka.attributeSelection.BestFirst, weka.attributeSelection.CfsSubsetEval]
//					[weka.attributeSelection.Ranker, weka.attributeSelection.GainRatioAttributeEval]
//					[weka.attributeSelection.GreedyStepwise, weka.attributeSelection.CfsSubsetEval]
//					[weka.attributeSelection.Ranker, weka.attributeSelection.InfoGainAttributeEval]
//					[weka.attributeSelection.Ranker, weka.attributeSelection.OneRAttributeEval]
//					[weka.attributeSelection.Ranker, weka.attributeSelection.PrincipalComponents]
//					[weka.attributeSelection.Ranker, weka.attributeSelection.ReliefFAttributeEval]
//					[weka.attributeSelection.Ranker, weka.attributeSelection.SymmetricalUncertAttributeEval]

		String search1 = "weka.attributeSelection.Ranker";
		String evaluation1 = "weka.attributeSelection.CorrelationAttributeEval";

		String search2 = "weka.attributeSelection.BestFirst";
		String evaluation2 = "weka.attributeSelection.CfsSubsetEval";

		String search3 = "weka.attributeSelection.Ranker";
		String evaluation3 = "weka.attributeSelection.GainRatioAttributeEval";

		String search4 = "weka.attributeSelection.GreedyStepwise";
		String evaluation4 = "weka.attributeSelection.CfsSubsetEval";

		String search5 = "weka.attributeSelection.Ranker";
		String evaluation5 = "weka.attributeSelection.InfoGainAttributeEval";

		String search6 = "weka.attributeSelection.Ranker";
		String evaluation6 = "weka.attributeSelection.OneRAttributeEval";

		String search7 = "weka.attributeSelection.Ranker";
		String evaluation7 = "weka.attributeSelection.PrincipalComponents";

		String search8 = "weka.attributeSelection.Ranker";
		String evaluation8 = "weka.attributeSelection.ReliefFAttributeEval";

		String search9 = "weka.attributeSelection.Ranker";
		String evaluation9 = "weka.attributeSelection.SymmetricalUncertAttributeEval";

		File jsonFile;
		try {
			jsonFile = Paths.get(WekaMLPlanWekaUtil.class.getClassLoader()
					.getResource(Paths.get("automl", "searchmodels", "weka", "weka-all-autoweka.json").toString())
					.toURI()).toFile();
			ComponentLoader cl = new ComponentLoader(jsonFile);
			Collection<Component> components = cl.getComponents();

			List<Pair<ASSearch, ASEvaluation>> evaluatorSearcherPairs = new ArrayList<Pair<ASSearch, ASEvaluation>>();
			List<Classifier> classifiers = new ArrayList<Classifier>();

			// add null entry for the case where no preprocessing is used
			evaluatorSearcherPairs.add(new Pair<ASSearch, ASEvaluation>(null,null));
			
			evaluatorSearcherPairs.add(new Pair<ASSearch, ASEvaluation>(ASSearch.forName(search1, new String[] {}),
					ASEvaluation.forName(evaluation1, new String[] {})));
			evaluatorSearcherPairs.add(new Pair<ASSearch, ASEvaluation>(ASSearch.forName(search2, new String[] {}),
					ASEvaluation.forName(evaluation2, new String[] {})));
			evaluatorSearcherPairs.add(new Pair<ASSearch, ASEvaluation>(ASSearch.forName(search3, new String[] {}),
					ASEvaluation.forName(evaluation3, new String[] {})));
			evaluatorSearcherPairs.add(new Pair<ASSearch, ASEvaluation>(ASSearch.forName(search4, new String[] {}),
					ASEvaluation.forName(evaluation4, new String[] {})));
			evaluatorSearcherPairs.add(new Pair<ASSearch, ASEvaluation>(ASSearch.forName(search5, new String[] {}),
					ASEvaluation.forName(evaluation5, new String[] {})));
			evaluatorSearcherPairs.add(new Pair<ASSearch, ASEvaluation>(ASSearch.forName(search6, new String[] {}),
					ASEvaluation.forName(evaluation6, new String[] {})));
			evaluatorSearcherPairs.add(new Pair<ASSearch, ASEvaluation>(ASSearch.forName(search7, new String[] {}),
					ASEvaluation.forName(evaluation7, new String[] {})));
			evaluatorSearcherPairs.add(new Pair<ASSearch, ASEvaluation>(ASSearch.forName(search8, new String[] {}),
					ASEvaluation.forName(evaluation8, new String[] {})));
			evaluatorSearcherPairs.add(new Pair<ASSearch, ASEvaluation>(ASSearch.forName(search9, new String[] {}),
					ASEvaluation.forName(evaluation9, new String[] {})));

			for (Component component : components) {
				if (component.getName().contains("classifier") && !component.getName().contains("Poly")) {
					Classifier c = AbstractClassifier.forName(component.getName(), new String[] {});
					classifiers.add(c);
				}
			}
			
			for(Pair<ASSearch, ASEvaluation> preprocessors : evaluatorSearcherPairs) {
				for(Classifier classifier : classifiers) {
					result.add(new MLPipeline(preprocessors.getFirst(), preprocessors.getSecond(), classifier));
				}
			}

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return result;
	}

}
