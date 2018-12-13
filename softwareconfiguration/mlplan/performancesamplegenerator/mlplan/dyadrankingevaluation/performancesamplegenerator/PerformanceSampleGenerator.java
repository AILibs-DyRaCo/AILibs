package mlplan.dyadrankingevaluation.performancesamplegenerator;


import java.io.File;
import java.util.concurrent.ThreadLocalRandom;

import de.upb.crc901.mlpipeline_evaluation.PerformanceDBAdapter;
import de.upb.crc901.mlplan.multiclass.wekamlplan.MLPlanWekaBuilder;
import de.upb.crc901.mlplan.multiclass.wekamlplan.MLPlanWekaClassifier;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.WEKAPipelineFactory;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.WekaMLPlanWekaClassifier;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import hasco.core.Util;
import hasco.model.ComponentInstance;
import jaicore.basic.SQLAdapter;
import jaicore.ml.cache.ReproducibleInstances;
import jaicore.ml.core.evaluation.measure.singlelabel.MultiClassPerformanceMeasure;
import jaicore.planning.graphgenerators.task.tfd.TFDNode;
import jaicore.search.algorithms.standard.bestfirst.nodeevaluation.INodeEvaluator;
import jaicore.search.algorithms.standard.random.RandomSearch;
import jaicore.search.core.interfaces.GraphGenerator;
import jaicore.search.model.other.SearchGraphPath;
import jaicore.search.model.probleminputs.GraphSearchInput;
import jaicore.search.model.travesaltree.Node;

public class PerformanceSampleGenerator {

	public static void main(String args[]) {
		PerformanceSampleGenerator psg = new PerformanceSampleGenerator();
		psg.evaluate();
	}
	
	public void evaluate() {
		try {
			SQLAdapter adapter = new SQLAdapter("host", "user", "password", "databse");
			PerformanceDBAdapter performanceDBAdapter = new PerformanceDBAdapter(adapter, "performance_stuff");
			MLPlanWekaBuilder builder = new MLPlanWekaBuilder(
					new File("conf/automl/searchmodels/weka/weka-all-autoweka.json"),
					new File("conf/automl/mlplan.properties"), MultiClassPerformanceMeasure.ERRORRATE,
					performanceDBAdapter);

			MLPlanWekaClassifier mlplan = new WekaMLPlanWekaClassifier(builder);
			
			ReproducibleInstances data = ReproducibleInstances.fromOpenML("181", "4350e421cdc16404033ef1812ea38c01");
			data.setCacheLookup(false);
			data.setCacheStorage(true);
			data.setClassIndex(data.numAttributes() - 1);
			mlplan.setLoggerName("mlplan");
			mlplan.setTimeout(60);
			mlplan.setData(data);
//			mlplan.setPreferredNodeEvaluator(this.new UninformedNodeEvaluator());
//			mlplan.buildClassifier(data);
			GraphGenerator gg = mlplan.getGraphGenerator();
			GraphSearchInput gsi = new GraphSearchInput(gg);
			RandomSearch rs = new RandomSearch(gsi, 0);
			while(rs.hasNext()) {
				SearchGraphPath sgp = rs.nextSolution();
//				System.out.println(sgp.toString());
				TFDNode goalNode = (TFDNode) sgp.getNodes().get(sgp.getNodes().size()-1);
				ComponentInstance ci = Util.getSolutionCompositionFromState(mlplan.getComponents(), goalNode.getState(), false);
				WEKAPipelineFactory factory = new WEKAPipelineFactory();
				MLPipeline mlp = factory.getComponentInstantiation(ci);
				System.out.println(mlp);
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private class UninformedNodeEvaluator implements INodeEvaluator<TFDNode, Double>{

		@Override
		public Double f(Node<TFDNode, ?> node) throws Exception {
			return 0.3;
		}
	}
}
