package de.upb.crc901.automl.metamining.pipelinecharacterizing;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.regex.Pattern;

import org.junit.Test;
import org.semanticweb.owlapi.model.OWLOntologyCreationException;

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import de.upb.crc901.mlplan.metamining.pipelinecharacterizing.ComponentInstanceStringConverter;
import de.upb.crc901.mlplan.metamining.pipelinecharacterizing.WEKAOntologyConnector;
import de.upb.crc901.mlplan.metamining.pipelinecharacterizing.WEKAPipelineCharacterizer;
import hasco.model.ComponentInstance;
import hasco.serialization.ComponentLoader;
import hasco.serialization.HASCOJacksonModule;
import junit.framework.Assert;
import treeminer.TreeMiner;

/**
 * This tests the functionalities of the
 * {@link ComponentInstanceStringConverter} which makes extensive use of the
 * provided {@link TreeMiner}.
 * 
 * In particular, this tests whether or not weka labels are correctly replaced
 * by integer values (mainly performance reasons). And if the TreeMiner
 * representations of Meta-Features are correctly serialized.
 * 
 * @author Mirko Jürgens
 *
 */
public class TreeMinerSerializerTest {
	
	String componentInstanceJSON = "{\"component\": {\"name\": \"weka.classifiers.meta.AdaBoostM1\", \"parameters\": [{\"name\": \"pActivator\", \"numeric\": false, \"categorical\": true, \"defaultValue\": \"0\", \"defaultDomain\": {\"values\": [\"0\", \"1\"]}}, {\"name\": \"Q\", \"numeric\": false, \"categorical\": true, \"defaultValue\": true, \"defaultDomain\": {\"values\": [\"true\", \"false\"]}}, {\"name\": \"S\", \"numeric\": false, \"categorical\": true, \"defaultValue\": \"1\", \"defaultDomain\": {\"values\": [\"1\"]}}, {\"name\": \"I\", \"numeric\": true, \"categorical\": false, \"defaultValue\": 10, \"defaultDomain\": {\"max\": 128, \"min\": 2, \"integer\": true}}, {\"name\": \"P\", \"numeric\": true, \"categorical\": false, \"defaultValue\": 100, \"defaultDomain\": {\"max\": 100, \"min\": 100, \"integer\": true}}], \"dependencies\": [{\"premise\": [[{\"x\": {\"name\": \"pActivator\", \"numeric\": false, \"categorical\": true, \"defaultValue\": \"0\", \"defaultDomain\": {\"values\": [\"0\", \"1\"]}}, \"y\": {\"values\": [\"0\"]}}]], \"conclusion\": [{\"x\": {\"name\": \"P\", \"numeric\": true, \"categorical\": false, \"defaultValue\": 100, \"defaultDomain\": {\"max\": 100, \"min\": 100, \"integer\": true}}, \"y\": {\"max\": 100, \"min\": 100, \"integer\": true}}]}, {\"premise\": [[{\"x\": {\"name\": \"pActivator\", \"numeric\": false, \"categorical\": true, \"defaultValue\": \"0\", \"defaultDomain\": {\"values\": [\"0\", \"1\"]}}, \"y\": {\"values\": [\"1\"]}}]], \"conclusion\": [{\"x\": {\"name\": \"P\", \"numeric\": true, \"categorical\": false, \"defaultValue\": 100, \"defaultDomain\": {\"max\": 100, \"min\": 100, \"integer\": true}}, \"y\": {\"max\": 100, \"min\": 50, \"integer\": true}}]}], \"providedInterfaces\": [\"weka.classifiers.meta.AdaBoostM1\", \"AbstractClassifier\", \"MetaClassifier\", \"BaseClassifier\"], \"requiredInterfaces\": {}}, \"parameterValues\": {\"I\": \"15\", \"P\": \"63\", \"Q\": \"true\", \"S\": \"1\", \"pActivator\": \"1\"}, \"satisfactionOfRequiredInterfaces\": {}, \"parametersThatHaveBeenSetExplicitly\": [{\"name\": \"pActivator\", \"numeric\": false, \"categorical\": true, \"defaultValue\": \"0\", \"defaultDomain\": {\"values\": [\"0\", \"1\"]}}, {\"name\": \"P\", \"numeric\": true, \"categorical\": false, \"defaultValue\": 100, \"defaultDomain\": {\"max\": 100, \"min\": 100, \"integer\": true}}, {\"name\": \"I\", \"numeric\": true, \"categorical\": false, \"defaultValue\": 10, \"defaultDomain\": {\"max\": 128, \"min\": 2, \"integer\": true}}, {\"name\": \"S\", \"numeric\": false, \"categorical\": true, \"defaultValue\": \"1\", \"defaultDomain\": {\"values\": [\"1\"]}}, {\"name\": \"Q\", \"numeric\": false, \"categorical\": true, \"defaultValue\": true, \"defaultDomain\": {\"values\": [\"true\", \"false\"]}}], \"parametersThatHaveNotBeenSetExplicitly\": []}";
	/*Obtained from training */
	private static final String expectedPatterns = "[100, 100 109 -1, 100 109 115 -1 -1, 100 109 115 147 -1 -1 -1, 100 109 115 147 148 -1 -1 -1 -1, 100 109 115 147 148 149 -1 -1 -1 -1 -1, 100 109 115 147 148 149 150 -1 -1 -1 -1 -1 -1, 100 109 115 147 148 149 150 151 -1 -1 -1 -1 -1 -1 -1, 100 109 115 147 148 149 150 151 152 -1 -1 -1 -1 -1 -1 -1 -1, 100 109 115 147 148 149 150 151 152 153 -1 -1 -1 -1 -1 -1 -1 -1 -1, 100 109 115 147 148 149 150 151 152 154 -1 -1 -1 -1 -1 -1 -1 -1 -1, 100 109 115 147 148 149 150 151 152 155 -1 -1 -1 -1 -1 -1 -1 -1 -1, 100 109 115 147 148 149 150 151 152 156 -1 -1 -1 -1 -1 -1 -1 -1 -1, 101, 101 81 -1, 102, 102 78 -1, 102 79 -1, 104, 104 114 -1, 104 114 128 -1 -1, 104 114 128 73 -1 -1 -1, 106, 106 123 -1, 106 123 93 -1 -1, 106 123 94 -1 -1, 106 123 96 -1 -1, 106 123 97 -1 -1, 106 123 98 -1 -1, 106 126 -1, 106 126 95 -1 -1, 107, 107 68 -1, 108, 108 107 -1, 108 107 68 -1 -1, 108 129 -1, 108 129 69 -1 -1, 108 129 71 -1 -1, 108 129 76 -1 -1, 109, 109 115 -1, 109 115 147 -1 -1, 109 115 147 148 -1 -1 -1, 109 115 147 148 149 -1 -1 -1 -1, 109 115 147 148 149 150 -1 -1 -1 -1 -1, 109 115 147 148 149 150 151 -1 -1 -1 -1 -1 -1, 109 115 147 148 149 150 151 152 -1 -1 -1 -1 -1 -1 -1, 109 115 147 148 149 150 151 152 153 -1 -1 -1 -1 -1 -1 -1 -1, 109 115 147 148 149 150 151 152 154 -1 -1 -1 -1 -1 -1 -1 -1, 109 115 147 148 149 150 151 152 155 -1 -1 -1 -1 -1 -1 -1 -1, 109 115 147 148 149 150 151 152 156 -1 -1 -1 -1 -1 -1 -1 -1, 110, 110 100 -1, 110 100 109 -1 -1, 110 100 109 115 -1 -1 -1, 110 100 109 115 147 -1 -1 -1 -1, 110 100 109 115 147 148 -1 -1 -1 -1 -1, 110 100 109 115 147 148 149 -1 -1 -1 -1 -1 -1, 110 100 109 115 147 148 149 150 -1 -1 -1 -1 -1 -1 -1, 110 100 109 115 147 148 149 150 151 -1 -1 -1 -1 -1 -1 -1 -1, 110 100 109 115 147 148 149 150 151 152 -1 -1 -1 -1 -1 -1 -1 -1 -1, 110 100 109 115 147 148 149 150 151 152 153 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 110 100 109 115 147 148 149 150 151 152 154 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 110 100 109 115 147 148 149 150 151 152 155 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 110 100 109 115 147 148 149 150 151 152 156 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 111, 111 101 -1, 111 101 81 -1 -1, 112, 112 104 -1, 112 104 114 -1 -1, 112 104 114 128 -1 -1 -1, 112 104 114 128 73 -1 -1 -1 -1, 112 119 -1, 112 119 108 -1 -1, 112 119 108 107 -1 -1 -1, 112 119 108 107 68 -1 -1 -1 -1, 112 119 108 129 -1 -1 -1, 112 119 108 129 69 -1 -1 -1 -1, 112 119 108 129 71 -1 -1 -1 -1, 112 119 108 129 76 -1 -1 -1 -1, 112 119 146 -1 -1, 112 119 146 67 -1 -1 -1, 112 119 72 -1 -1, 112 119 75 -1 -1, 112 158 -1, 112 158 74 -1 -1, 112 159 -1, 112 159 160 -1 -1, 112 159 160 66 -1 -1 -1, 112 159 160 70 -1 -1 -1, 113, 113 106 -1, 113 106 123 -1 -1, 113 106 123 93 -1 -1 -1, 113 106 123 94 -1 -1 -1, 113 106 123 96 -1 -1 -1, 113 106 123 97 -1 -1 -1, 113 106 123 98 -1 -1 -1, 113 106 126 -1 -1, 113 106 126 95 -1 -1 -1, 113 121 -1, 113 121 83 -1 -1, 113 121 83 157 -1 -1 -1, 113 121 83 157 157 -1 -1 -1 -1, 113 121 83 157 157 110 -1 -1 -1 -1 -1, 113 121 83 157 157 110 100 -1 -1 -1 -1 -1 -1, 113 121 83 157 157 110 100 109 -1 -1 -1 -1 -1 -1 -1, 113 121 83 157 157 110 100 109 115 -1 -1 -1 -1 -1 -1 -1 -1, 113 121 83 157 157 110 100 109 115 147 -1 -1 -1 -1 -1 -1 -1 -1 -1, 113 121 83 157 157 110 100 109 115 147 148 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 113 121 83 157 157 110 100 109 115 147 148 149 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 113 121 83 157 157 110 100 109 115 147 148 149 150 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 113 121 83 157 157 110 100 109 115 147 148 149 150 151 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 113 121 83 157 157 110 100 109 115 147 148 149 150 151 152 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 113 121 83 157 157 110 100 109 115 147 148 149 150 151 152 153 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 113 121 83 157 157 110 100 109 115 147 148 149 150 151 152 154 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 113 121 83 157 157 110 100 109 115 147 148 149 150 151 152 155 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 113 121 83 157 157 110 100 109 115 147 148 149 150 151 152 156 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 113 125 -1, 113 125 89 -1 -1, 113 125 90 -1 -1, 113 125 91 -1 -1, 113 125 92 -1 -1, 113 130 -1, 113 130 102 -1 -1, 113 130 102 78 -1 -1 -1, 113 130 102 79 -1 -1 -1, 113 130 118 -1 -1, 113 130 118 77 -1 -1 -1, 114, 114 128 -1, 114 128 73 -1 -1, 115, 115 147 -1, 115 147 148 -1 -1, 115 147 148 149 -1 -1 -1, 115 147 148 149 150 -1 -1 -1 -1, 115 147 148 149 150 151 -1 -1 -1 -1 -1, 115 147 148 149 150 151 152 -1 -1 -1 -1 -1 -1, 115 147 148 149 150 151 152 153 -1 -1 -1 -1 -1 -1 -1, 115 147 148 149 150 151 152 154 -1 -1 -1 -1 -1 -1 -1, 115 147 148 149 150 151 152 155 -1 -1 -1 -1 -1 -1 -1, 115 147 148 149 150 151 152 156 -1 -1 -1 -1 -1 -1 -1, 116, 116 86 -1, 116 87 -1, 118, 118 77 -1, 119, 119 108 -1, 119 108 107 -1 -1, 119 108 107 68 -1 -1 -1, 119 108 129 -1 -1, 119 108 129 69 -1 -1 -1, 119 108 129 71 -1 -1 -1, 119 108 129 76 -1 -1 -1, 119 146 -1, 119 146 67 -1 -1, 119 72 -1, 119 75 -1, 121, 121 83 -1, 121 83 157 -1 -1, 121 83 157 157 -1 -1 -1, 121 83 157 157 110 -1 -1 -1 -1, 121 83 157 157 110 100 -1 -1 -1 -1 -1, 121 83 157 157 110 100 109 -1 -1 -1 -1 -1 -1, 121 83 157 157 110 100 109 115 -1 -1 -1 -1 -1 -1 -1, 121 83 157 157 110 100 109 115 147 -1 -1 -1 -1 -1 -1 -1 -1, 121 83 157 157 110 100 109 115 147 148 -1 -1 -1 -1 -1 -1 -1 -1 -1, 121 83 157 157 110 100 109 115 147 148 149 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 121 83 157 157 110 100 109 115 147 148 149 150 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 121 83 157 157 110 100 109 115 147 148 149 150 151 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 121 83 157 157 110 100 109 115 147 148 149 150 151 152 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 121 83 157 157 110 100 109 115 147 148 149 150 151 152 153 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 121 83 157 157 110 100 109 115 147 148 149 150 151 152 154 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 121 83 157 157 110 100 109 115 147 148 149 150 151 152 155 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 121 83 157 157 110 100 109 115 147 148 149 150 151 152 156 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 122, 122 113 -1, 122 113 106 -1 -1, 122 113 106 123 -1 -1 -1, 122 113 106 123 93 -1 -1 -1 -1, 122 113 106 123 94 -1 -1 -1 -1, 122 113 106 123 96 -1 -1 -1 -1, 122 113 106 123 97 -1 -1 -1 -1, 122 113 106 123 98 -1 -1 -1 -1, 122 113 106 126 -1 -1 -1, 122 113 106 126 95 -1 -1 -1 -1, 122 113 121 -1 -1, 122 113 121 83 -1 -1 -1, 122 113 121 83 157 -1 -1 -1 -1, 122 113 121 83 157 157 -1 -1 -1 -1 -1, 122 113 121 83 157 157 110 -1 -1 -1 -1 -1 -1, 122 113 121 83 157 157 110 100 -1 -1 -1 -1 -1 -1 -1, 122 113 121 83 157 157 110 100 109 -1 -1 -1 -1 -1 -1 -1 -1, 122 113 121 83 157 157 110 100 109 115 -1 -1 -1 -1 -1 -1 -1 -1 -1, 122 113 121 83 157 157 110 100 109 115 147 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 122 113 121 83 157 157 110 100 109 115 147 148 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 122 113 121 83 157 157 110 100 109 115 147 148 149 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 122 113 121 83 157 157 110 100 109 115 147 148 149 150 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 122 113 121 83 157 157 110 100 109 115 147 148 149 150 151 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 122 113 121 83 157 157 110 100 109 115 147 148 149 150 151 152 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 122 113 121 83 157 157 110 100 109 115 147 148 149 150 151 152 153 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 122 113 121 83 157 157 110 100 109 115 147 148 149 150 151 152 154 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 122 113 121 83 157 157 110 100 109 115 147 148 149 150 151 152 155 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 122 113 121 83 157 157 110 100 109 115 147 148 149 150 151 152 156 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 122 113 125 -1 -1, 122 113 125 89 -1 -1 -1, 122 113 125 90 -1 -1 -1, 122 113 125 91 -1 -1 -1, 122 113 125 92 -1 -1 -1, 122 113 130 -1 -1, 122 113 130 102 -1 -1 -1, 122 113 130 102 78 -1 -1 -1 -1, 122 113 130 102 79 -1 -1 -1 -1, 122 113 130 118 -1 -1 -1, 122 113 130 118 77 -1 -1 -1 -1, 122 116 -1, 122 116 86 -1 -1, 122 116 87 -1 -1, 122 124 -1, 122 124 111 -1 -1, 122 124 111 101 -1 -1 -1, 122 124 111 101 81 -1 -1 -1 -1, 122 124 85 -1 -1, 122 127 -1, 122 127 80 -1 -1, 122 127 84 -1 -1, 122 132 -1, 122 132 133 -1 -1, 122 132 134 -1 -1, 122 132 135 -1 -1, 122 132 136 -1 -1, 122 132 137 -1 -1, 122 132 138 -1 -1, 122 132 139 -1 -1, 122 132 140 -1 -1, 122 132 142 -1 -1, 123, 123 93 -1, 123 94 -1, 123 96 -1, 123 97 -1, 123 98 -1, 124, 124 111 -1, 124 111 101 -1 -1, 124 111 101 81 -1 -1 -1, 124 85 -1, 125, 125 89 -1, 125 90 -1, 125 91 -1, 125 92 -1, 126, 126 95 -1, 127, 127 80 -1, 127 84 -1, 128, 128 73 -1, 129, 129 69 -1, 129 71 -1, 129 76 -1, 130, 130 102 -1, 130 102 78 -1 -1, 130 102 79 -1 -1, 130 118 -1, 130 118 77 -1 -1, 132, 132 133 -1, 132 134 -1, 132 135 -1, 132 136 -1, 132 137 -1, 132 138 -1, 132 139 -1, 132 140 -1, 132 142 -1, 133, 134, 135, 136, 137, 138, 139, 140, 142, 146, 146 67 -1, 147, 147 148 -1, 147 148 149 -1 -1, 147 148 149 150 -1 -1 -1, 147 148 149 150 151 -1 -1 -1 -1, 147 148 149 150 151 152 -1 -1 -1 -1 -1, 147 148 149 150 151 152 153 -1 -1 -1 -1 -1 -1, 147 148 149 150 151 152 154 -1 -1 -1 -1 -1 -1, 147 148 149 150 151 152 155 -1 -1 -1 -1 -1 -1, 147 148 149 150 151 152 156 -1 -1 -1 -1 -1 -1, 148, 148 149 -1, 148 149 150 -1 -1, 148 149 150 151 -1 -1 -1, 148 149 150 151 152 -1 -1 -1 -1, 148 149 150 151 152 153 -1 -1 -1 -1 -1, 148 149 150 151 152 154 -1 -1 -1 -1 -1, 148 149 150 151 152 155 -1 -1 -1 -1 -1, 148 149 150 151 152 156 -1 -1 -1 -1 -1, 149, 149 150 -1, 149 150 151 -1 -1, 149 150 151 152 -1 -1 -1, 149 150 151 152 153 -1 -1 -1 -1, 149 150 151 152 154 -1 -1 -1 -1, 149 150 151 152 155 -1 -1 -1 -1, 149 150 151 152 156 -1 -1 -1 -1, 150, 150 151 -1, 150 151 152 -1 -1, 150 151 152 153 -1 -1 -1, 150 151 152 154 -1 -1 -1, 150 151 152 155 -1 -1 -1, 150 151 152 156 -1 -1 -1, 151, 151 152 -1, 151 152 153 -1 -1, 151 152 154 -1 -1, 151 152 155 -1 -1, 151 152 156 -1 -1, 152, 152 153 -1, 152 154 -1, 152 155 -1, 152 156 -1, 153, 154, 155, 156, 157, 157 157 -1, 157 157 110 -1 -1, 157 157 110 100 -1 -1 -1, 157 157 110 100 109 -1 -1 -1 -1, 157 157 110 100 109 115 -1 -1 -1 -1 -1, 157 157 110 100 109 115 147 -1 -1 -1 -1 -1 -1, 157 157 110 100 109 115 147 148 -1 -1 -1 -1 -1 -1 -1, 157 157 110 100 109 115 147 148 149 -1 -1 -1 -1 -1 -1 -1 -1, 157 157 110 100 109 115 147 148 149 150 -1 -1 -1 -1 -1 -1 -1 -1 -1, 157 157 110 100 109 115 147 148 149 150 151 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 157 157 110 100 109 115 147 148 149 150 151 152 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 157 157 110 100 109 115 147 148 149 150 151 152 153 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 157 157 110 100 109 115 147 148 149 150 151 152 154 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 157 157 110 100 109 115 147 148 149 150 151 152 155 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 157 157 110 100 109 115 147 148 149 150 151 152 156 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 158, 158 74 -1, 159, 159 160 -1, 159 160 66 -1 -1, 159 160 70 -1 -1, 160, 160 66 -1, 160 70 -1, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 83 157 -1, 83 157 157 -1 -1, 83 157 157 110 -1 -1 -1, 83 157 157 110 100 -1 -1 -1 -1, 83 157 157 110 100 109 -1 -1 -1 -1 -1, 83 157 157 110 100 109 115 -1 -1 -1 -1 -1 -1, 83 157 157 110 100 109 115 147 -1 -1 -1 -1 -1 -1 -1, 83 157 157 110 100 109 115 147 148 -1 -1 -1 -1 -1 -1 -1 -1, 83 157 157 110 100 109 115 147 148 149 -1 -1 -1 -1 -1 -1 -1 -1 -1, 83 157 157 110 100 109 115 147 148 149 150 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 83 157 157 110 100 109 115 147 148 149 150 151 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 83 157 157 110 100 109 115 147 148 149 150 151 152 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 83 157 157 110 100 109 115 147 148 149 150 151 152 153 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 83 157 157 110 100 109 115 147 148 149 150 151 152 154 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 83 157 157 110 100 109 115 147 148 149 150 151 152 155 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 83 157 157 110 100 109 115 147 148 149 150 151 152 156 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1, 84, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98]"; 
	
	
	private static final String serializedAlgorithm = "0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0";  

	ObjectMapper mapper;

	ComponentInstance cI;

	ComponentLoader loader;

	List<String> disallowedClassifiers = Arrays.asList("weka.classifiers.meta.MultiClassClassifier", "weka.classifiers.meta.LogitBoost", "weka.classifiers.functions.supportVector.Puk");
	public TreeMinerSerializerTest() throws JsonParseException, JsonMappingException, IOException, URISyntaxException, OWLOntologyCreationException {
		mapper = new ObjectMapper();
		mapper.registerModule(new HASCOJacksonModule());
		cI = mapper.readValue(componentInstanceJSON, ComponentInstance.class);
		File jsonFile = Paths.get(getClass().getClassLoader()
				.getResource(Paths.get("automl", "searchmodels", "weka", "weka-all-autoweka.json").toString()).toURI())
				.toFile();

		loader = new ComponentLoader(jsonFile);
		ComponentInstanceStringConverter converter = new ComponentInstanceStringConverter(new WEKAOntologyConnector(),Arrays.asList(cI), loader.getParamConfigs());
		converter.run();
		System.out.println(converter.getConvertedPipelines());
	}

	/**
	 * Tests if the ComponentInstanceStringConverter works with the double labels
	 * @throws OWLOntologyCreationException
	 */
	@Test
	public void testStringConverterConvertSingleInstance() throws OWLOntologyCreationException {
		ComponentInstanceStringConverter stringConverter = new ComponentInstanceStringConverter(
				new WEKAOntologyConnector(), Arrays.asList(cI), loader.getParamConfigs());
		Pattern pattern = Pattern.compile(" ");
		System.out.println(stringConverter.makeStringTreeRepresentation(cI));
		// check whether all patterns only contain doubles; will throw NumberFormatException otherwise
		pattern.splitAsStream(stringConverter.makeStringTreeRepresentation(cI)).allMatch(s-> s == "-" || Double.parseDouble(s) > 0);
	}

	/**
	 * Checks if the WekaAlgorithm miner can load the patterns from the provided file.
	 * @throws URISyntaxException
	 * @throws JsonParseException
	 * @throws JsonMappingException
	 * @throws IOException
	 * @throws InterruptedException
	 */
	@Test
	public void testTreeMinerLoadPreComputedAlgorithmPatterns()
			throws URISyntaxException, JsonParseException, JsonMappingException, IOException, InterruptedException {
		WEKAPipelineCharacterizer characterizer = new WEKAPipelineCharacterizer(loader.getParamConfigs());
		characterizer.buildFromFile();
		Assert.assertEquals(expectedPatterns, characterizer.getFoundPipelinePatterns().toString());
	}
	
	@Test
	public void testSingleCharacterization() {
		WEKAPipelineCharacterizer characterizer = new WEKAPipelineCharacterizer(loader.getParamConfigs());
		characterizer.buildFromFile();
		double[] characterization = characterizer.characterize(cI);
		//check if this is a correct vector
		Assert.assertTrue(Arrays.stream(characterization).allMatch(d -> d == 0.0 || d == 1.0));

		double [] unserializedArray = Pattern.compile(" ").splitAsStream(serializedAlgorithm).mapToDouble(Double::parseDouble).toArray();
		Assert.assertTrue(Arrays.equals(unserializedArray, characterization));

		
	}

}
