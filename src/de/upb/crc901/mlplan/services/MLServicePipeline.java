package de.upb.crc901.mlplan.services;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.upb.crc901.mlplan.services.MLPipelinePlan.MLPipe;
import de.upb.crc901.mlplan.services.MLPipelinePlan.WekaAttributeSelectionPipe;
import de.upb.crc901.services.core.EasyClient;
import de.upb.crc901.services.core.EnvironmentState;
import de.upb.crc901.services.core.HttpServiceServer;
import de.upb.crc901.services.core.OntologicalTypeMarshallingSystem;
import de.upb.crc901.services.core.ServiceCompositionResult;
import de.upb.crc901.services.core.ServiceHandle;
import jaicore.ml.WekaUtil;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class MLServicePipeline implements Classifier, Serializable {

	private long expirationDate;
	private boolean trained;
	private int timeForTrainingPipeline;
	private DescriptiveStatistics timesForPrediction;
	private final MLPipelinePlan constructionPlan;
	
	private static final Logger logger = LoggerFactory.getLogger(MLServicePipeline.class);

	private final EnvironmentState servicesContainer = new EnvironmentState();

	private transient OntologicalTypeMarshallingSystem otms = new OntologicalTypeMarshallingSystem();

	private final List<String> PPFieldNames = new LinkedList<>();

	private final String classifierFieldName;

	private final int getSecondsRemaining() {
		return this.expirationDate > 0 ? (int) (this.expirationDate - System.currentTimeMillis()) / 1000 : Integer.MAX_VALUE;
	}

	public MLServicePipeline(final MLPipelinePlan plan) throws InterruptedException {
		super();
		if (!plan.isValid()) {
			throw new RuntimeException("The given plan is not valid..");
		}
		constructionPlan = plan;

		int varIDCounter = 1;
		EasyClient constructorEC = new EasyClient().withOTMS(this.otms);

		// build composition
		for (MLPipe attrPipe : plan.getAttrSelections()) {
			String ppFieldName = "service" + varIDCounter;
			this.PPFieldNames.add(ppFieldName);

			// set host of the attribute selection
			constructorEC.withHost(attrPipe.getHost());

			if (attrPipe instanceof WekaAttributeSelectionPipe) {
				WekaAttributeSelectionPipe wekaASPipe = (WekaAttributeSelectionPipe) attrPipe;
				String searcherFieldName = "searcher" + varIDCounter;
				String searchOptions = "searcherOptions" + varIDCounter;

				String evalFieldName = "eval" + varIDCounter;
				String evalOptions = "evalOptions" + varIDCounter;

				// add inputs from attribute selection
				constructorEC.withKeywordArgument(searcherFieldName, wekaASPipe.getSearcher()).withKeywordArgument(evalFieldName, wekaASPipe.getEval())

						.withKeywordArgument(searchOptions, wekaASPipe.getSearcherOptions()).withKeywordArgument(evalOptions, wekaASPipe.getEvalOptions());

				// add a construction line to the composition
				constructorEC.withAddedConstructOperation(ppFieldName, // output field name of the created servicehandle
						attrPipe.getName(), // classpath of the preprocessor
						searcherFieldName, searchOptions, evalFieldName, evalOptions);
			} else {
				constructorEC.withKeywordArgument("asOptions" + varIDCounter, attrPipe.getOptions());
				// add a construction line to the composition
				constructorEC.withAddedConstructOperation(ppFieldName, // output field name of the created servicehandle
						attrPipe.getName(), // classpath of the preprocessor
						"asOptions" + varIDCounter);

			}

			varIDCounter++;
		}
		this.classifierFieldName = "service" + varIDCounter;
		String classifierOptions = "classifierOptions";

		// set host for classifier
		constructorEC.withHost(plan.getClassifierPipe().getHost());

		// add a stringlist for classifier
		constructorEC.withKeywordArgument(classifierOptions, plan.getClassifierPipe().getOptions());

		constructorEC.withAddedConstructOperation(this.classifierFieldName, // output field name of the created servicehandle
				plan.getClassifierPipe().getName(), // classpath of the classifier
				classifierOptions); // no args for the classifier construct

		// set the host to the first service:
		if (!plan.getAttrSelections().isEmpty()) {
			constructorEC.withHost(plan.getAttrSelections().get(0).getHost());
		}

		try {
			// send server request:
			// System.out.println("Sending the following construct composition:\n" +
			// constructorEC.getCurrentCompositionText());

			ServiceCompositionResult result = constructorEC.dispatch();
			this.servicesContainer.extendBy(result); // add the services to out state
		} catch (IOException e) {
			logger.error("Error in the creation of the services involved in a MLServicePipeline. Exception: {}. Message: {}", e.getClass().getName(), e.getMessage());
		}

		// Service creation done!

		// for (String fieldname : servicesContainer.serviceHandleFieldNames()) {
		// ServiceHandle sh = (ServiceHandle) servicesContainer.retrieveField(fieldname).getData();
		// System.out.println("\t" + fieldname + ":=" + sh.getServiceAddress());
		// }

		// check if all our preprocessors and the classifier is created:
		for (String ppFieldName : this.PPFieldNames) {
			if (!this.servicesContainer.containsField(ppFieldName) || // if it isn't returned by the server
					!(this.servicesContainer.retrieveField(ppFieldName).getData() instanceof ServiceHandle) || // or if it isn't a servicehandle
					!((ServiceHandle) this.servicesContainer.retrieveField(ppFieldName).getData()).isSerialized()) // of if it doesn't contain an id.
			{
				throw new RuntimeException(ppFieldName);
			}
		}
		// same check for the classifier:
		if (!this.servicesContainer.containsField(this.classifierFieldName) || // if it isn't returned by the server
				!(this.servicesContainer.retrieveField(this.classifierFieldName).getData() instanceof ServiceHandle) || // or if it isn't a servicehandle
				!((ServiceHandle) this.servicesContainer.retrieveField(this.classifierFieldName).getData()).isSerialized()) // of if it doesn't contain an id.
		{
			throw new RuntimeException(this.classifierFieldName);
		}
	}

	@Override
	public void buildClassifier(final Instances data) throws Exception {

		int secondsRemaining = this.getSecondsRemaining();
		if (secondsRemaining < 5) {
			throw new IllegalStateException("Cannot train, only " + secondsRemaining + " lifetime remain!");
		}

		int invocationNumber = 1;
		String dataInFieldName = "i1";
		EasyClient trainEC = new EasyClient().withOTMS(this.otms).withInputs(this.servicesContainer).withPositionalArgument(data);
		// trainEC.withClientID(Thread.currentThread().getId() + "");
		// create train composition
		for (String ppFieldname : this.PPFieldNames) {
			String dataOutFieldName = "data" + invocationNumber;
			trainEC.withAddedMethodOperation("Empty", ppFieldname, "train", dataInFieldName);
			trainEC.withAddedMethodOperation(dataOutFieldName, ppFieldname, "preprocess", dataInFieldName);
			// output of this pipe is the input of the next one:
			dataInFieldName = dataOutFieldName;
			invocationNumber++;
		}
		trainEC.withAddedMethodOperation("empty", this.classifierFieldName, "train", dataInFieldName);

		long start = System.currentTimeMillis();
		try {
			// System.out.println("Sending the following train composition:\n " +
			// trainEC.getCurrentCompositionText());
			// send train request:
			trainEC.dispatch();
		} catch (IOException ex) {
			ex.printStackTrace();
		}
		this.timeForTrainingPipeline = (int) (System.currentTimeMillis() - start);
		this.trained = true;
		this.timesForPrediction = new DescriptiveStatistics();
	}

	@Override
	public double classifyInstance(final Instance instance) throws Exception {
		Instances instances = new Instances(instance.dataset(), 1);
		instances.add(instance);
		return this.classifyInstances(instances)[0];
	}

	public double[] classifyInstances(final Instances instances) throws Exception {

		int secondsRemaining = this.getSecondsRemaining();
		if (secondsRemaining < 5) {
			throw new IllegalStateException("Cannot train, only " + secondsRemaining + " lifetime remain!");
		}

		int invocationNumber = 1;
		String dataInFieldName = "i1";

		EasyClient predictEC = new EasyClient().withOTMS(this.otms).withInputs(this.servicesContainer).withPositionalArgument(instances); // translates to
																																			// i1

		// create train composition
		for (String ppFieldname : this.PPFieldNames) {
			String dataOutFieldName = "data" + invocationNumber;
			predictEC.withAddedMethodOperation(dataOutFieldName, ppFieldname, "preprocess", dataInFieldName);
			// out put of this pipe it the input of the next one:
			dataInFieldName = dataOutFieldName;
			// create output name for the next data
			invocationNumber++;
		}
		predictEC.withAddedMethodOperation("predictions", this.classifierFieldName, "predict", dataInFieldName);

		long start = System.currentTimeMillis();
		ServiceCompositionResult result;
		try {
			// System.out.println("Sending the following predict composition:\n " +
			// predictEC.getCurrentCompositionText());
			// send predict request:
			result = predictEC.dispatch();
		} catch (IOException ex) {
			// ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		long end = System.currentTimeMillis();
		timesForPrediction.addValue(end-start);

		@SuppressWarnings("unchecked")
		List<String> predictedLabels = (List<String>) result.get("predictions").getData();
		double[] predictedIndices = new double[predictedLabels.size()];
		for (int i = 0, size = predictedIndices.length; i < size; i++) {
			predictedIndices[i] = instances.classAttribute().indexOfValue(predictedLabels.get(i));
		}
		return predictedIndices;
	}

	@Override
	public double[] distributionForInstance(final Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}

	private void readObject(final java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
		in.defaultReadObject();
		this.otms = new OntologicalTypeMarshallingSystem();
	}

	public static void main(final String[] args) throws Throwable {
		HttpServiceServer server = null;
		try {
			int jasePort = 8000;
			server = HttpServiceServer.TEST_SERVER();
			// create a ml pp plan:
			MLPipelinePlan plan = new MLPipelinePlan();
			// add attribute selections:

			String jaseHost = "localhost:" + jasePort;
			String paseHost = "localhost:" + 5000;
			//
			plan.onHost(paseHost).addAttributeSelection("sklearn.preprocessing.Imputer").addOptions("-strategy \"median\"", "-axis 0");

			plan.onHost(jaseHost);

			// plan.addWekaAttributeSelection() .withSearcher("weka.attributeSelection.Ranker")
			// .withEval("weka.attributeSelection.CorrelationAttributeEval");
			// plan.addWekaAttributeSelection() .withSearcher("weka.attributeSelection.Ranker")
			// .withEval("weka.attributeSelection.PrincipalComponents");
			plan.onHost(paseHost).addAttributeSelection("sklearn.feature_selection.VarianceThreshold");

			plan.onHost(jaseHost).setClassifier("weka.classifiers.functions.MultilayerPerceptron");
			// plan.onHost(paseHost);
			// plan.setClassifier("tflib.NeuralNet").addOptions("-layer_count 2", "-epochs 1000",
			// "-learning_rate 0.3");

			// System.out.println("Create MLServicePipeline with classifier and "
			// + plan.getAttrSelections().size() + " many preprocessors.");

			MLServicePipeline pl = new MLServicePipeline(plan);

			Instances wekaInstances = new Instances(
					new BufferedReader(new FileReader("../CrcTaskBasedConfigurator/testrsc" + File.separator + "polychotomous" + File.separator + "audiology.arff")));
			wekaInstances.setClassIndex(wekaInstances.numAttributes() - 1);
			// System.out.println(wekaInstances.numClasses());
			List<Instances> split = WekaUtil.getStratifiedSplit(wekaInstances, new Random(0), .7f);

			pl.buildClassifier(split.get(0));
			System.out.println("Building done!");

			int mistakes = 0;
			int index = 0;
			double[] predictions = pl.classifyInstances(split.get(1));
			for (Instance instance : split.get(1)) {
				double prediction = predictions[index];
				if (instance.classValue() != prediction) {
					mistakes++;
				}
				index++;
			}
			System.out.println("Pipeline done. This many mistakes were made:" + mistakes + ". Error rate: " + (mistakes * 1f / split.get(1).size()));
		} finally {
			if (server != null) {
				server.shutdown();
			}
		}

	}

	public List<String> getPPFieldNames() {
		return this.PPFieldNames;
	}

	public String getClassifierFieldName() {
		return this.classifierFieldName;
	}

	public int getTimeForTrainingPipeline() {
		return this.timeForTrainingPipeline;
	}

	public DescriptiveStatistics getTimesForPrediction() {
		return this.timesForPrediction;
	}

	public long getExpirationDate() {
		return this.expirationDate;
	}

	public void setExpirationDate(final long expirationDate) {
		this.expirationDate = expirationDate;
	}
	

	public MLPipelinePlan getConstructionPlan() {
		return constructionPlan;
	}

	@Override
	public String toString() {
		return constructionPlan.toString();
	}
	
	
}
