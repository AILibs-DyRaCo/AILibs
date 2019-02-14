package de.upb.crc901.mlplan.test;

import java.io.IOException;
import java.util.List;

import org.junit.Test;

import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.WekaMLPlanWekaUtil;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import jaicore.ml.cache.ReproducibleInstances;
import jaicore.ml.core.evaluation.measure.singlelabel.ZeroOneLoss;
import jaicore.ml.evaluation.evaluators.weka.MonteCarloCrossValidationEvaluator;
import jaicore.ml.evaluation.evaluators.weka.SimpleEvaluatorMeasureBridge;
import weka.classifiers.Classifier;

public class WekaMLPlanWekaUtilTest {

	@Test
	public void test() {
		List<MLPipeline> allLegalPipelines = WekaMLPlanWekaUtil.getAllLegalWekaPipelinesWithDefaultConfig();
		for(MLPipeline pl : allLegalPipelines)
			System.out.println(pl.toString());
		System.out.println("number pipelines: " + allLegalPipelines.size());
		
		ReproducibleInstances data;
		try {
			data = ReproducibleInstances.fromOpenML("40983", "4350e421cdc16404033ef1812ea38c01");
			MonteCarloCrossValidationEvaluator evaluator = new MonteCarloCrossValidationEvaluator(new SimpleEvaluatorMeasureBridge(new ZeroOneLoss()), 5, data, 0.7d, 54);
			Classifier cl = allLegalPipelines.get(27);
			double loss = evaluator.evaluate(cl);
			System.out.println("Pipeline " + cl.toString() + " has a MCCV 0/1 loss of " + loss);
		} catch (NumberFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}

}
