package defaultEval.core;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Scanner;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import hasco.model.BooleanParameterDomain;
import hasco.model.CategoricalParameterDomain;
import hasco.model.Component;
import hasco.model.ComponentInstance;
import hasco.model.NumericParameterDomain;
import hasco.model.Parameter;
import hasco.model.ParameterDomain;
import hasco.serialization.ComponentLoader;
import scala.annotation.elidable;

public class SMACOptimizer extends Optimizer{
	

	public SMACOptimizer(Component searcher, Component evaluator, Component classifier, String dataSet, File environment, int seed) {
		super(searcher, evaluator, classifier, dataSet, environment, seed);
	}

	@Override
	public void optimize() {
		
		generatePCSFile();
		
		generatePyWrapper();
		
		// start SMAC
		try {
			Runtime rt = Runtime.getRuntime();
			
			
			System.out.println(environment.getAbsolutePath() + "/optimizer/smac/smac.bat "+
									"--run-obj QUALITY " + 
									"--use-instances false "+
							"--numberOfRunsLimit 10 "+
							"--seed " + seed + " "+
							"--pcs-file "+environment.getAbsolutePath()+"/pcs/"+buildFileName()+".pcs "+
							"--output-dir " +environment.getAbsolutePath() + "/smac-output/"+buildFileName()+" " +
							"--algo \"python "+environment.getAbsolutePath()+"/py_wrapper/"+buildFileName()+".py\"\" "
					);
			
			Process proc = rt.exec(
							environment.getAbsolutePath() + "/optimizer/smac/smac.bat "+
									"--run-obj QUALITY " + 
									"--use-instances false "+
							"--numberOfRunsLimit 10 "+
							"--seed " + seed + " "+
							"--pcs-file "+environment.getAbsolutePath()+"/pcs/"+buildFileName()+".pcs "+
							"--output-dir " +environment.getAbsolutePath() + "/smac-output/"+buildFileName()+" " +
							"--algo \"python "+environment.getAbsolutePath()+"/py_wrapper/"+buildFileName()+".py\"\" "
							);
			
			InputStream i = proc.getInputStream();
			int r = 0;
			while ((r = i.read()) != -1) {
				System.out.write(r);
			}
			int exitValue = proc.exitValue();
			
			createFinalInstances();
			
			
		} catch (IOException e) {
			// TODO
			e.printStackTrace();
		}
		
		System.out.println("final-searcher: " + finalSearcher);
		System.out.println("final-evaluator: " + finalEvaluator);
		System.out.println("final-classifier: " + finalClassifier);
	}

	private void createFinalInstances() throws IOException, FileNotFoundException {
		// read outputs
		File outputRunsAndResults = new File(environment.getAbsolutePath() + "/smac-output/"+buildFileName() + "/NoScenarioFile/state-run" + seed + "/runs_and_results-it1.csv");
		File outputUniqConfigurations = new File(environment.getAbsolutePath() + "/smac-output/"+buildFileName() + "/NoScenarioFile/state-run" + seed + "/uniq_configurations-it1.csv");
		
		CSVParser parserRunsAndResults = new CSVParser(new FileReader(outputRunsAndResults), CSVFormat.DEFAULT.withHeader());
		CSVParser parserUniqConfigurations = new CSVParser(new FileReader(outputUniqConfigurations), CSVFormat.DEFAULT);
		
		// find opt
		int bestIndex = 0;
		double bestScore = 100;
		for (CSVRecord result : parserRunsAndResults) {
			try {
				double newScore = Double.valueOf(result.get(3));
				if(newScore < bestScore) {
					bestScore = newScore;
					bestIndex = Integer.valueOf(result.get(0));
				}
			} catch (NumberFormatException e) {
				//skip
			}
		}
		
		for (CSVRecord resultConfig : parserUniqConfigurations) {
			if(Integer.valueOf(resultConfig.get(0)) == bestIndex) {
				int index = 1;
				HashMap<String, String> searcherParameter = new HashMap<>();
				HashMap<String, String> evaluatorParameter = new HashMap<>();
				HashMap<String, String> classifierParameter = new HashMap<>();
				
				if(searcher != null) {
					// get preprocessor config
					for (Parameter parameter : searcher.getParameters()) {
						searcherParameter.put(parameter.getName(), resultConfig.get(index++));
					}
					for (Parameter parameter : evaluator.getParameters()) {
						evaluatorParameter.put(parameter.getName(), resultConfig.get(index++));
					}
				}
				for (Parameter parameter : classifier.getParameters()) {
					classifierParameter.put(parameter.getName(), resultConfig.get(index++));
				}
				
				finalSearcher = new ComponentInstance(searcher, searcherParameter, new HashMap<>());
				finalEvaluator = new ComponentInstance(evaluator, evaluatorParameter, new HashMap<>());
				finalClassifier = new ComponentInstance(classifier, classifierParameter, new HashMap<>());
			}
		}
		parserRunsAndResults.close();
		parserUniqConfigurations.close();
	}
	
	
	private void generatePyWrapper() {
		// generate py-wrapper file
		PrintStream pyWrapperStream = null;
		// TODO do not generate if allready there
		
		try {
			pyWrapperStream = new PrintStream(new FileOutputStream(new File(environment.getAbsolutePath() + "/py_wrapper/" + buildFileName() + ".py")));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		pyWrapperStream.println("#!/usr/bin/python");
		pyWrapperStream.println("import sys, math");
		pyWrapperStream.println("from subprocess import call");
		
		for (Parameter parameter : parameterList) {
			pyWrapperStream.println( parameter.getName() + " = " + getInitialValue(parameter.getDefaultDomain()));
		}
		
		pyWrapperStream.println("for i in range(len(sys.argv)-1): ");
		int i = 0;
		for (Parameter parameter : parameterList) {
			if(i == 0) {
				pyWrapperStream.println("\t if (sys.argv[i] == '-" + parameter.getName() + "'):");
				
			}else {
				pyWrapperStream.println("\t elif(sys.argv[i] == '-" + parameter.getName() + "'):");
			}
			
			pyWrapperStream.println("\t \t " + parameter.getName() + " = "+getConverter(parameter.getDefaultDomain())+"(sys.argv[i+1])");
			i++;
		}
		
		// run java programm
		pyWrapperStream.print("call(\"java -jar "+environment.getAbsolutePath()+"/PipelineEvaluator.jar");
		
		pyWrapperStream.print(" " + dataSet);
		
		
		if(searcher != null) {
			pyWrapperStream.print(" " + searcher.getName());
			
			for (Parameter parameter : searcher.getParameters()) {
				pyWrapperStream.print(" \"+"+createDomainWrapper(parameter.getName(), parameter.getDefaultDomain())+" + \""); // TODO only for floats 
			}
			
			pyWrapperStream.print(" " + evaluator.getName());
			
			for (Parameter parameter : evaluator.getParameters()) {
				pyWrapperStream.print(" \"+ "+createDomainWrapper(parameter.getName(), parameter.getDefaultDomain())+" + \""); // TODO only for floats 
			}
			
		}else {
			pyWrapperStream.print(" null");
		}
		
		pyWrapperStream.print(" " + classifier.getName());
		
		for (Parameter parameter : classifier.getParameters()) {
			pyWrapperStream.print(" \"+"+createDomainWrapper(parameter.getName(), parameter.getDefaultDomain())+" + \"");
		}
		
		
		pyWrapperStream.print(" " + environment.getAbsolutePath()+"/results/" + buildFileName() + ".txt");
		
		pyWrapperStream.println("\")");
		
		
		// read result file
		
		pyWrapperStream.println("file = open(\""+environment.getAbsolutePath()+"\\\\results\\\\"+buildFileName()+".txt\", \"r\")");
		pyWrapperStream.println("yValue = float(file.read())");
		
		
		pyWrapperStream.println("print \"Result for SMAC: SUCCESS, 0, 0, %f, 0\" % yValue");
		
		pyWrapperStream.close();
	}

	private void generatePCSFile() {
		// generate params file
		PrintStream pcsStream = null;
		try {
			pcsStream = new PrintStream(new FileOutputStream(new File(environment.getAbsolutePath()+"/pcs/"+buildFileName()+".pcs")));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		// TODO Namensueberschneidungen?
		for (Parameter p : parameterList) {
			ParameterDomain pd = p.getDefaultDomain();

			// Numeric (integer or real/double)
			if(pd instanceof NumericParameterDomain) {
				NumericParameterDomain n_pd = (NumericParameterDomain) pd;
				pcsStream.println(p.getName() + " " + (n_pd.isInteger() ? "integer" : "real") + " [" + n_pd.getMin() + ", " +n_pd.getMax() + "] [" + p.getDefaultValue().toString() + "]" );
			}
			
			// Boolean (categorical)
			else if(pd instanceof BooleanParameterDomain) {
				BooleanParameterDomain b_pd = (BooleanParameterDomain) pd;
				pcsStream.println(p.getName() + " categorical {true, false} [" + p.getDefaultValue().toString() + "]" );
			}
			
			//categorical
			else if(pd instanceof CategoricalParameterDomain) {
				CategoricalParameterDomain c_pd = (CategoricalParameterDomain) pd;
				pcsStream.print(p.getName() + " categorical {");
				for (int i = 0; i < c_pd.getValues().length; i++) {
					pcsStream.print(c_pd.getValues()[i]);
					if(i != c_pd.getValues().length - 1) {
						pcsStream.print(",");
					}
				}
				pcsStream.println("} [" + p.getDefaultValue().toString() + "]");
			}
		}
		
		pcsStream.close();
	}
	
	
	private String buildFileName() {
		return (searcher != null) ? (searcher.getName()+"_"+evaluator.getName()) : "null" + "_" + classifier.getName() + "_" + dataSet;
	}
	
	private String getConverter(ParameterDomain pd) {
		if(pd instanceof NumericParameterDomain) {
			NumericParameterDomain n_pd = (NumericParameterDomain) pd;
			return n_pd.isInteger() ? "int" : "float";
		}
		return "";
	}
	
	private String createDomainWrapper(String input, ParameterDomain pd) {
		if(pd instanceof NumericParameterDomain) {
			NumericParameterDomain n_pd = (NumericParameterDomain) pd;
			return n_pd.isInteger() ? input : "\"{:.9f}\".format("+input+")";
		}
		
		return input;
	}
	
	
	
	public static void main(String[] args) {
		
		ComponentLoader cl_p = new ComponentLoader();
		ComponentLoader cl_c = new ComponentLoader();
		
		try {
			Util.loadClassifierComponents(cl_c);
			Util.loadPreprocessorComponents(cl_p);	
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		Component searcher = null;
		Component evaluator = null;
		Component classifier = null;
		
		for (Component c : cl_p.getComponents()) {
			if(c.getName().equals("weka.attributeSelection.BestFirst")) {
				searcher = c;
			}
		}
		
		for (Component c : cl_p.getComponents()) {
			if(c.getName().equals("weka.attributeSelection.CorrelationAttributeEval")) {
				evaluator = c;
			}
		}
		
		for (Component c : cl_c.getComponents()) {
			if(c.getName().equals("weka.classifiers.functions.Logistic")) {
				classifier = c;
			}
		}
		
		SMACOptimizer o = new SMACOptimizer(searcher, evaluator, classifier, "breast-cancer", new File("F:\\Data\\Uni\\PG\\DefaultEvalEnvironment"), 0);
		o.optimize();
		
		
	}
	
	
	private String getInitialValue(ParameterDomain pd) {
		if(pd instanceof NumericParameterDomain) {
			return "0";
		}
		
		// Boolean (categorical)
		else if(pd instanceof BooleanParameterDomain) {
			return "'true'";
		}
		
		//categorical
		else if(pd instanceof CategoricalParameterDomain) {
			return "''";
		}
		
		return "0";
	}
	
	
	
	
}
