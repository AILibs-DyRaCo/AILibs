package de.upb.crc901.mlplan.search.evaluators;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apache.commons.math.stat.descriptive.SynchronizedMultivariateSummaryStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.upb.crc901.mlplan.core.CodePlanningUtil;
import de.upb.crc901.mlplan.core.MLPipeline;
import de.upb.crc901.mlplan.core.MLUtil;
import de.upb.crc901.mlplan.core.SolutionEvaluator;
import de.upb.crc901.mlplan.core.SuvervisedFilterSelector;
import jaicore.logic.fol.structure.Literal;
import jaicore.ml.WekaUtil;
import jaicore.planning.graphgenerators.task.tfd.TFDNode;
import jaicore.planning.model.ceoc.CEOCAction;
import jaicore.search.algorithms.standard.core.NodeAnnotationEvent;
import jaicore.search.algorithms.standard.core.SolutionAnnotationEvent;
import jaicore.search.algorithms.standard.core.SolutionFoundEvent;
import jaicore.search.structure.core.Node;

public class BalancedRandomCompletionEvaluator extends RandomCompletionEvaluator<Double> {

	private final static Logger logger = LoggerFactory.getLogger(BalancedRandomCompletionEvaluator.class);
	private static final List<String> classifierRanking = Arrays.asList(new String[] { "OneR", "RandomTree", "J48", "IBk", "NaiveBayesMultinomial", "NaiveBayes", "RandomForest",
			"SimpleLogistic", "MultiLayerPerceptron", "VotedPerceptron", "SMO", "Logistic" });

	private final Map<String, Integer> regionCounter = new HashMap<>();

	public BalancedRandomCompletionEvaluator(Random random, int samples, SolutionEvaluator evaluator) {
		super(random, samples, evaluator);
	}

	@Override
	public Double computeEvaluationPriorToCompletion(Node<TFDNode, ?> n, List<TFDNode> path, List<CEOCAction> plan, List<String> currentProgram) throws Exception {

		/* determine intended/chosen classifier */
		String intentendClassifierName = getIntendedClassifier(n);
		if (intentendClassifierName != null) {
			String simpleName = intentendClassifierName.substring(intentendClassifierName.lastIndexOf(".") + 1);
			if (!classifierRanking.contains(simpleName))
				return classifierRanking.size() * 2.0;

			/* determine chosen preprocessor */
			String preprocessor = CodePlanningUtil.getPreprocessorEvaluatorFromPipelineGenerationCode(currentProgram);
			int offset = (preprocessor.equals("")) ? 0 : classifierRanking.size();
			return offset + classifierRanking.indexOf(simpleName) * 1.0;
		} else {

			String classifierName = getNameOfClassifier(n, currentProgram);

			/* if the classifier has not been decided, return 0.0 */
			if (classifierName == null)
				return 0.0;

			/* check whether a filter and a classifier have been defined */
			getSolutionEventBus().post(new NodeAnnotationEvent<>(n.getPoint(), "classifier", classifierName));

			/* if the classifier has just been defined, check for the standard configuration */
			boolean classifierJustDefined = currentProgram.get(currentProgram.size() - 1).contains("new " + classifierName);
			if (classifierJustDefined) {
				String preprocessor = CodePlanningUtil.getPreprocessorEvaluatorFromPipelineGenerationCode(currentProgram);
				String plName = preprocessor + "&" + classifierName.substring(classifierName.lastIndexOf(".") + 1);
				logger.info("Classifier has just been chosen, now try its standard configuration.");
				try {
					
					/* check whether oneR has been set */
					String reference = preprocessor + "&OneR";
					if (plFails.containsKey(reference)) {
						return null;
					}
					
					long start = System.currentTimeMillis();
					Integer f = evaluator.getSolutionScore(MLUtil.extractPipelineFromPlan(plan));
					int fTime = (int)(System.currentTimeMillis() - start);
					if (f != null) {
						fValues.put(n, (double) f);
					} else {
						fValues.put(n, Double.MAX_VALUE);
					}

					/* return the value right now, because we have no completion for this node */
					if (f == null)
						return null;
					getSolutionEventBus().post(new SolutionFoundEvent<TFDNode,Double>(path, (double)f));
					getSolutionEventBus().post(new SolutionAnnotationEvent<>(path, "fTime", fTime));
					return f != null ? (double) f : null;
				} catch (InterruptedException e) {
					
					/* punish preprocessor for timeout */
					int fails = 0;
					if (!ppFails.containsKey(preprocessor)) {
						fails = 1;
					} else
						fails = ppFails.get(preprocessor) + 1;
					ppFails.put(preprocessor, fails);
					
					/* punish pipeline prototype */
					fails = 0;
					if (!plFails.containsKey(plName)) {
						fails = 1;
					} else
						fails = plFails.get(plName) + 1;
					plFails.put(plName, fails);
					throw e;
				}
			}

			/* this is the case where we really start computing f-values by simulation */
			return null;
		}
	}

	@Override
	protected Double convertErrorRateToNodeEvaluation(Integer errorRate) {
		return errorRate != null ? errorRate * 1.0 : null;
	}

	@Override
	protected double getExpectedUpperBoundForRelativeDistanceToOptimalSolution(Node<TFDNode, ?> n, List<TFDNode> path, List<CEOCAction> plan, List<String> currentProgram) {
		String nameOfClassifier = getNameOfClassifier(n, currentProgram);
		if (nameOfClassifier == null)
			return 1;

		MLPipeline pl = null;
		try {
			pl = MLUtil.extractPipelineFromPlan(plan);
		} catch (Exception e) {
			return 1;
		}

		/* determine actually set params */
		List<CEOCAction> paramSettingActions = plan.stream().filter(a -> a.getOperation().getName().toLowerCase().contains("param")).collect(Collectors.toList());
		int numOfParamSettingActions = paramSettingActions.size();

		/* determine possible params */
		int numOfParamsSettable = 0;
		try {
			numOfParamsSettable += WekaUtil.getOptionsOfWekaAlgorithm(pl.getBaseClassifier()).size();
			SuvervisedFilterSelector preprocessor = pl.getPreprocessors().isEmpty() ? null : pl.getPreprocessors().get(0);
			if (preprocessor != null) {
				numOfParamsSettable += WekaUtil.getOptionsOfWekaAlgorithm(preprocessor.getSearcher()).size();
				numOfParamsSettable += WekaUtil.getOptionsOfWekaAlgorithm(preprocessor.getEvaluator()).size();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return numOfParamSettingActions == 0 ? 1 : 1f / numOfParamSettingActions;
	}

	private String getNameOfClassifier(Node<TFDNode, ?> n, List<String> currentProgram) {
		Pattern p = Pattern.compile("new (weka\\.classifiers\\.[a-zA-Z0-9\\.]*)");
		Optional<String> classifierLine = currentProgram.stream().filter(line -> p.matcher(line).find()).findAny();
		if (classifierLine.isPresent()) {
			String line = classifierLine.get();
			Matcher m = p.matcher(line);
			m.find();
			return m.group(1);
		}
		logger.info("No classifier defined yet, so returning 0.");
		return null;
	}

	private String getIntendedClassifier(Node<TFDNode, ?> n) {
		Pattern p = Pattern.compile("(weka\\.classifiers\\.[a-zA-Z0-9\\.]*):__construct");
		for (Literal l : n.getPoint().getRemainingTasks()) {
			Matcher m = p.matcher(l.getPropertyName());
			if (m.find()) {
				return m.group(1);
			}
		}
		return null;
	}

	private String getIntendedPreprocessor(Node<TFDNode, ?> n) {
		Pattern p = Pattern.compile("(weka\\.attributeSelection\\.[a-zA-Z0-9\\.]*):__construct");
		String combo = "";
		for (Literal l : n.getPoint().getRemainingTasks()) {
			Matcher m = p.matcher(l.getPropertyName());
			if (m.find()) {
				if (combo.length() > 0)
					combo += "&";
				combo += m.group(1);
			}
		}
		return combo.length() > 0 ? combo : null;
	}
}
