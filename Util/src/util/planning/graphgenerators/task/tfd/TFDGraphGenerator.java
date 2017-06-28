package util.planning.graphgenerators.task.tfd;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import util.logic.Literal;
import util.logic.Monom;
import util.planning.graphgenerators.task.TaskPlannerUtil;
import util.planning.model.core.Action;
import util.planning.model.core.Operation;
import util.planning.model.task.stn.MethodInstance;
import util.planning.model.task.stn.STNPlanningProblem;
import util.search.core.NodeExpansionDescription;
import util.search.core.NodeType;
import util.search.core.GraphGenerator;
import util.search.graphgenerator.GoalTester;
import util.search.graphgenerator.RootGenerator;
import util.search.graphgenerator.SuccessorGenerator;

public class TFDGraphGenerator implements GraphGenerator<TFDNode,String> {

	private final STNPlanningProblem problem;
	private final Map<String,Operation> primitiveTasks = new HashMap<>();

	public TFDGraphGenerator(STNPlanningProblem problem) {
		this.problem = problem;
		for (Operation op : problem.getDomain().getOperations())
			primitiveTasks.put(op.getName(), op);
	}

	@Override
	public RootGenerator<TFDNode> getRootGenerator() {
		return () -> Arrays.asList(new TFDNode[]{new TFDNode(problem.getInit(), TaskPlannerUtil.getTaskChainOfTotallyOrderedNetwork(problem.getNetwork()))});
	}

	@Override
	public SuccessorGenerator<TFDNode,String> getSuccessorGenerator() {
		return l -> {
			List<NodeExpansionDescription<TFDNode,String>> successors = new ArrayList<>();
			Monom state = l.getPoint().getState();
			List<Literal> currentlyRemainingTasks = new ArrayList<>(l.getPoint().getRemainingTasks());
			Literal nextTaskTmp = currentlyRemainingTasks.get(0);
			currentlyRemainingTasks.remove(0);
			String nextTaskName = nextTaskTmp.getPropertyName().substring(nextTaskTmp.getPropertyName().indexOf("-") + 1, nextTaskTmp.getPropertyName().length());
			Literal nextTask = new Literal(nextTaskName, nextTaskTmp.getParameters());
			
			/* if the task is primitive */
			if (primitiveTasks.containsKey(nextTask.getPropertyName())) {
				
				for (Action applicableAction : TaskPlannerUtil.getActionsForPrimitiveTaskThatAreApplicableInState(null, primitiveTasks.get(nextTask.getPropertyName()), nextTask, state)) {
					Monom stateCopy = new Monom(state);
					TFDNodeUtil.updateState(stateCopy, applicableAction);
					successors.add(new NodeExpansionDescription<>(l.getPoint(), new TFDNode(stateCopy, currentlyRemainingTasks, null, applicableAction), "edge label", NodeType.OR));
				}
			}
			
			/* otherwise determine methods for the task */
			else {
				for (MethodInstance instance : TaskPlannerUtil.getMethodInstancesForTaskThatAreApplicableInState(null, this.problem.getDomain().getMethods(), nextTask, state)) {
					
					/* derive remaining network for this instance */
					List<Literal> remainingTasks = TaskPlannerUtil.getTaskChainOfTotallyOrderedNetwork(instance.getNetwork());
					remainingTasks.addAll(currentlyRemainingTasks);
					successors.add(new NodeExpansionDescription<>(l.getPoint(), new TFDNode(state, remainingTasks, instance, null), "edge label", NodeType.OR));
				}
			}
			return successors;
		};
	}

	@Override
	public GoalTester<TFDNode> getGoalTester() {
		return l -> l.getPoint().getRemainingTasks().isEmpty();
	}
}
