package de.upb.crc901.mlplan.test;

import java.io.IOException;
import java.util.List;

import org.junit.Test;

import de.upb.crc901.mlplan.multiclass.wekamlplan.MLPlanWekaClassifier;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.WekaMLPlanWekaClassifier;
import jaicore.ml.cache.ReproducibleInstances;
import jaicore.planning.graphgenerators.task.tfd.TFDNode;
import jaicore.search.core.interfaces.GraphGenerator;
import jaicore.search.model.travesaltree.NodeExpansionDescription;
import jaicore.search.structure.graphgenerator.SingleRootGenerator;

public class WekaMLPlanWekaUtilTest {

	@Test
	public void test() {
//		List<MLPipeline> allLegalPipelines = WekaMLPlanWekaUtil.getAllLegalWekaPipelinesWithDefaultConfig();
//		for(MLPipeline pl : allLegalPipelines)
//			System.out.println(pl.toString());
//		System.out.println("number pipelines: " + allLegalPipelines.size());
//		
//		ReproducibleInstances data;
//		try {
//			MonteCarloCrossValidationEvaluator evaluator = new MonteCarloCrossValidationEvaluator(new SimpleEvaluatorMeasureBridge(new ZeroOneLoss()), 5, data, 0.7d, 54);
//			Classifier cl = allLegalPipelines.get(27);
//			double loss = evaluator.evaluate(cl);
//			System.out.println("Pipeline " + cl.toString() + " has a MCCV 0/1 loss of " + loss);
//		} catch (NumberFormatException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		} catch (Exception e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}

		try {
			ReproducibleInstances data = ReproducibleInstances.fromOpenML("40983", "4350e421cdc16404033ef1812ea38c01");
			MLPlanWekaClassifier mlplan = new WekaMLPlanWekaClassifier();
			mlplan.setData(data);
			GraphGenerator gg = mlplan.getGraphGenerator();
			SingleRootGenerator<TFDNode> srg = (SingleRootGenerator<TFDNode>) gg.getRootGenerator();
			List<NodeExpansionDescription<?, ?>> successors = gg.getSuccessorGenerator().generateSuccessors(srg.getRoot());
			System.out.println(successors.size());
			for(NodeExpansionDescription<?, ?> succ : successors)
				System.out.println("From: " + succ.getFrom().toString() + " to: " + succ.getTo().toString());
		} catch (IOException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}

}
