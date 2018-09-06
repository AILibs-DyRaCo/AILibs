package jaicore.search.algorithms.standard.bestfirst;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Queue;
import java.util.Set;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

import org.aeonbits.owner.ConfigFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.eventbus.Subscribe;

import jaicore.basic.ILoggingCustomizable;
import jaicore.basic.algorithm.AlgorithmEvent;
import jaicore.basic.algorithm.AlgorithmExecutionCanceledException;
import jaicore.basic.algorithm.AlgorithmFinishedEvent;
import jaicore.basic.algorithm.AlgorithmInitializedEvent;
import jaicore.basic.algorithm.AlgorithmState;
import jaicore.basic.algorithm.IAlgorithmConfig;
import jaicore.concurrent.TimeoutTimer;
import jaicore.concurrent.TimeoutTimer.TimeoutSubmitter;
import jaicore.graphvisualizer.events.graphEvents.GraphInitializedEvent;
import jaicore.graphvisualizer.events.graphEvents.NodeParentSwitchEvent;
import jaicore.graphvisualizer.events.graphEvents.NodeReachedEvent;
import jaicore.graphvisualizer.events.graphEvents.NodeRemovedEvent;
import jaicore.graphvisualizer.events.graphEvents.NodeTypeSwitchEvent;
import jaicore.logging.LoggerUtil;
import jaicore.search.algorithms.standard.AbstractORGraphSearch;
import jaicore.search.algorithms.standard.bestfirst.events.GraphSearchSolutionCandidateFoundEvent;
import jaicore.search.algorithms.standard.bestfirst.events.NodeAnnotationEvent;
import jaicore.search.algorithms.standard.bestfirst.events.NodeExpansionCompletedEvent;
import jaicore.search.algorithms.standard.bestfirst.events.NodeExpansionJobSubmittedEvent;
import jaicore.search.algorithms.standard.bestfirst.events.SolutionAnnotationEvent;
import jaicore.search.algorithms.standard.bestfirst.events.SuccessorComputationCompletedEvent;
import jaicore.search.algorithms.standard.bestfirst.model.OpenCollection;
import jaicore.search.algorithms.standard.bestfirst.model.PriorityQueueOpen;
import jaicore.search.algorithms.standard.bestfirst.nodeevaluation.DecoratingNodeEvaluator;
import jaicore.search.algorithms.standard.bestfirst.nodeevaluation.ICancelableNodeEvaluator;
import jaicore.search.algorithms.standard.bestfirst.nodeevaluation.IGraphDependentNodeEvaluator;
import jaicore.search.algorithms.standard.bestfirst.nodeevaluation.INodeEvaluator;
import jaicore.search.algorithms.standard.bestfirst.nodeevaluation.ISolutionReportingNodeEvaluator;
import jaicore.search.core.interfaces.GraphGenerator;
import jaicore.search.model.other.EvaluatedSearchGraphPath;
import jaicore.search.model.probleminputs.GeneralEvaluatedTraversalTree;
import jaicore.search.model.travesaltree.GraphEventBus;
import jaicore.search.model.travesaltree.Node;
import jaicore.search.model.travesaltree.NodeExpansionDescription;
import jaicore.search.structure.graphgenerator.MultipleRootGenerator;
import jaicore.search.structure.graphgenerator.NodeGoalTester;
import jaicore.search.structure.graphgenerator.PathGoalTester;
import jaicore.search.structure.graphgenerator.RootGenerator;
import jaicore.search.structure.graphgenerator.SingleRootGenerator;
import jaicore.search.structure.graphgenerator.SuccessorGenerator;

public class BestFirst<I extends GeneralEvaluatedTraversalTree<N, A, V>, N, A, V extends Comparable<V>> extends AbstractORGraphSearch<I, EvaluatedSearchGraphPath<N, A, V>, N, A, V, Node<N, V>, A> implements ILoggingCustomizable {

	public enum ParentDiscarding {
		NONE, OPEN, ALL
	}

	/* algorithm configuration */
	private IAlgorithmConfig config;

	/* algorithm state and statistics */
	private int createdCounter;
	private int expandedCounter;
	private boolean initialized = false;
	private boolean interrupted = false;
	private boolean canceled = false;
	private Timer timeoutTimer;
	private AlgorithmState state = AlgorithmState.created;
	private EvaluatedSearchGraphPath<N, A, V> bestSeenSolution;
	private List<NodeExpansionDescription<N, A>> lastExpansion = new ArrayList<>();

	/* communication */
	protected final GraphEventBus<Node<N, V>> graphEventBus = new GraphEventBus<>();
	private Logger logger = LoggerFactory.getLogger(BestFirst.class);
	protected final Map<N, Node<N, V>> ext2int = new ConcurrentHashMap<>();

	/* search related objects */
	protected OpenCollection<Node<N, V>> open = new PriorityQueueOpen<>();
	private final Set<N> expanded = new HashSet<>();
	protected final GraphGenerator<N, A> graphGenerator;
	protected final RootGenerator<N> rootGenerator;
	protected final SuccessorGenerator<N, A> successorGenerator;
	protected final boolean checkGoalPropertyOnEntirePath;
	protected final PathGoalTester<N> pathGoalTester;
	protected final NodeGoalTester<N> nodeGoalTester;
	private ParentDiscarding parentDiscarding;

	/* computation of f */
	protected final INodeEvaluator<N, V> nodeEvaluator;
	private final boolean solutionReportingNodeEvaluator;
	private INodeEvaluator<N, V> timeoutNodeEvaluator;
	private TimeoutSubmitter timeoutSubmitter;
	private int timeoutForComputationOfF;

	protected final Queue<EvaluatedSearchGraphPath<N, A, V>> solutions = new LinkedBlockingQueue<>();
	protected final Queue<GraphSearchSolutionCandidateFoundEvent<N, A, V>> pendingSolutionFoundEvents = new LinkedBlockingQueue<>();

	/* parallelization */
	protected int additionalThreadsForExpansion = 0;
	private Semaphore fComputationTickets;
	private ExecutorService pool;
	protected final AtomicInteger activeJobs = new AtomicInteger(0);
	private Semaphore nodeModelSemaphore = new Semaphore(1);

	/* Synchronization of open queue modifications */
	private Lock openLock = new ReentrantLock(); // lock that has to be locked before accessing the open queue
	private Condition addedNodeToOpenCondition = this.openLock.newCondition(); // condition that is signaled whenever a node is added to the open queue

	private class NodeBuilder implements Runnable {

		private final Collection<N> todoList;
		private final Node<N, V> expandedNodeInternal;
		private final NodeExpansionDescription<N, A> successorDescription;

		public NodeBuilder(final Collection<N> todoList, final Node<N, V> expandedNodeInternal, final NodeExpansionDescription<N, A> successorDescription) {
			super();
			this.todoList = todoList;
			this.expandedNodeInternal = expandedNodeInternal;
			this.successorDescription = successorDescription;
		}

		private void communicateJobFinished() {
			synchronized (this.todoList) {
				this.todoList.remove(this.successorDescription.getTo());
				if (this.todoList.isEmpty()) {
					BestFirst.this.graphEventBus.post(new NodeExpansionCompletedEvent<>(this.expandedNodeInternal));
				}
			}
		}

		@Override
		public void run() {
			try {
				if (BestFirst.this.canceled || BestFirst.this.interrupted) {
					this.communicateJobFinished();
					return;
				}
				BestFirst.this.logger.debug("Start node creation.");
				BestFirst.this.lastExpansion.add(this.successorDescription);

				Node<N, V> newNode = BestFirst.this.newNode(this.expandedNodeInternal, this.successorDescription.getTo());

				/* update creation counter */
				BestFirst.this.createdCounter++;

				/* set timeout on thread that interrupts it after the timeout */
				int taskId = -1;
				if (BestFirst.this.timeoutForComputationOfF > 0) {
					if (BestFirst.this.timeoutSubmitter == null) {
						BestFirst.this.timeoutSubmitter = TimeoutTimer.getInstance().getSubmitter();
					}
					taskId = BestFirst.this.timeoutSubmitter.interruptMeAfterMS(BestFirst.this.timeoutForComputationOfF);
				}

				/* compute node label */
				V label = null;
				boolean computationTimedout = false;
				long startComputation = System.currentTimeMillis();
				try {
					label = BestFirst.this.nodeEvaluator.f(newNode);
					if (BestFirst.this.canceled || BestFirst.this.interrupted) {
						this.communicateJobFinished();
						return;
					}

					/* check whether the required time exceeded the timeout */
					long fTime = System.currentTimeMillis() - startComputation;
					if (BestFirst.this.timeoutForComputationOfF > 0 && fTime > BestFirst.this.timeoutForComputationOfF + 1000) {
						BestFirst.this.logger.warn("Computation of f for node {} took {}ms, which is more than the allowed {}ms", newNode, fTime, BestFirst.this.timeoutForComputationOfF);
					}
				} catch (InterruptedException e) {
					BestFirst.this.logger.debug("Received interrupt during computation of f.");
					BestFirst.this.graphEventBus.post(new NodeTypeSwitchEvent<>(newNode, "or_timedout"));
					newNode.setAnnotation("fError", "Timeout");
					computationTimedout = true;
					try {
						label = BestFirst.this.timeoutNodeEvaluator != null ? BestFirst.this.timeoutNodeEvaluator.f(newNode) : null;
					} catch (Throwable e2) {
						e2.printStackTrace();
					}
				} catch (Throwable e) {
					BestFirst.this.logger.error("Observed an exception during computation of f:\n{}", LoggerUtil.getExceptionInfo(e));
					newNode.setAnnotation("fError", e);
					BestFirst.this.graphEventBus.post(new NodeTypeSwitchEvent<>(newNode, "or_ffail"));
				}
				if (taskId >= 0) {
					BestFirst.this.timeoutSubmitter.cancelTimeout(taskId);
				}

				/* register time required to compute this node label */
				long fTime = System.currentTimeMillis() - startComputation;
				newNode.setAnnotation("fTime", fTime);

				/* if no label was computed, prune the node and cancel the computation */
				if (label == null) {
					if (!computationTimedout) {
						BestFirst.this.logger.info("Not inserting node {} since its label is missing!", newNode);
					} else {
						BestFirst.this.logger.info("Not inserting node {} because computation of f-value timed out.", newNode);
					}
					if (!newNode.getAnnotations().containsKey("fError")) {
						newNode.setAnnotation("fError", "f-computer returned NULL");
					}
					BestFirst.this.graphEventBus.post(new NodeTypeSwitchEvent<>(newNode, "or_pruned"));
					if (BestFirst.this.pool != null) {
						BestFirst.this.activeJobs.decrementAndGet();
						BestFirst.this.fComputationTickets.release();
					}
					this.communicateJobFinished();
					return;
				}
				newNode.setInternalLabel(label);

				BestFirst.this.logger.info("Inserting successor {} of {} to OPEN. F-Value is {}", newNode, this.expandedNodeInternal, label);
				// assert !open.contains(newNode) && !expanded.contains(newNode.getPoint()) :
				// "Inserted node is already in OPEN or even expanded!";

				/* if we discard (either only on OPEN or on both OPEN and CLOSED) */
				boolean nodeProcessed = false;
				if (BestFirst.this.parentDiscarding != ParentDiscarding.NONE) {

					BestFirst.this.openLock.lock();
					try {
						/* determine whether we already have the node AND it is worse than the one we want to insert */
						Optional<Node<N, V>> existingIdenticalNodeOnOpen = BestFirst.this.open.stream().filter(n -> n.getPoint().equals(newNode.getPoint())).findFirst();
						if (existingIdenticalNodeOnOpen.isPresent()) {
							Node<N, V> existingNode = existingIdenticalNodeOnOpen.get();
							if (newNode.compareTo(existingNode) < 0) {
								BestFirst.this.graphEventBus.post(new NodeTypeSwitchEvent<>(newNode, "or_" + (newNode.isGoal() ? "solution" : "open")));
								BestFirst.this.graphEventBus.post(new NodeRemovedEvent<>(existingNode));
								BestFirst.this.open.remove(existingNode);
								BestFirst.this.open.add(newNode);
							} else {
								BestFirst.this.graphEventBus.post(new NodeRemovedEvent<>(newNode));
							}
							nodeProcessed = true;
						}

						/* if parent discarding is not only for OPEN but also for CLOSE (and the node was not on OPEN), check the list of expanded nodes */
						else if (BestFirst.this.parentDiscarding == ParentDiscarding.ALL) {

							/* reopening, if the node is already on CLOSED */
							Optional<N> existingIdenticalNodeOnClosed = BestFirst.this.expanded.stream().filter(n -> n.equals(newNode.getPoint())).findFirst();
							if (existingIdenticalNodeOnClosed.isPresent()) {
								Node<N, V> node = BestFirst.this.ext2int.get(existingIdenticalNodeOnClosed.get());
								if (newNode.compareTo(node) < 0) {
									node.setParent(newNode.getParent());
									node.setInternalLabel(newNode.getInternalLabel());
									BestFirst.this.expanded.remove(node.getPoint());
									BestFirst.this.open.add(node);
									BestFirst.this.graphEventBus.post(new NodeParentSwitchEvent<Node<N, V>>(node, node.getParent(), newNode.getParent()));
								}
								BestFirst.this.graphEventBus.post(new NodeRemovedEvent<Node<N, V>>(newNode));
								nodeProcessed = true;
							}
						}
					} finally {
						BestFirst.this.openLock.unlock();
					}
				}

				/* if parent discarding is turned off OR if the node was node processed by a parent discarding rule, just insert it on OPEN */
				if (!nodeProcessed) {

					if (!newNode.isGoal()) {
						BestFirst.this.openLock.lock();
						try {
							BestFirst.this.open.add(newNode);
						} finally {
							BestFirst.this.openLock.unlock();
						}
					}
					BestFirst.this.graphEventBus.post(new NodeTypeSwitchEvent<>(newNode, "or_" + (newNode.isGoal() ? "solution" : "open")));
					BestFirst.this.createdCounter++;
				}

				/* Recognize solution in cache together with annotation */
				if (newNode.isGoal()) {
					EvaluatedSearchGraphPath<N, A, V> solution = new EvaluatedSearchGraphPath<>(BestFirst.this.getTraversalPath(newNode), null, newNode.getInternalLabel());

					/* if the node evaluator has not reported the solution already anyway, register the solution */
					if (!BestFirst.this.solutionReportingNodeEvaluator) {
						BestFirst.this.registerSolutionCandidateViaEvent(new GraphSearchSolutionCandidateFoundEvent<>(solution));
					}
				}

				/* free resources if this is computed by helper threads */
				if (BestFirst.this.pool != null) {
					BestFirst.this.activeJobs.decrementAndGet();
					BestFirst.this.fComputationTickets.release();
				}
			} catch (Throwable e) {
				e.printStackTrace();
			}
			this.communicateJobFinished();
		}
	}

	public BestFirst(final I problem) {
		this(problem, ConfigFactory.create(IBestFirstConfig.class));
	}

	@SuppressWarnings("unchecked")
	public BestFirst(final I problem, final IBestFirstConfig config) {
		super(problem);
		this.graphGenerator = problem.getGraphGenerator();
		this.rootGenerator = this.graphGenerator.getRootGenerator();
		this.successorGenerator = this.graphGenerator.getSuccessorGenerator();
		this.checkGoalPropertyOnEntirePath = !(this.graphGenerator.getGoalTester() instanceof NodeGoalTester);
		if (this.checkGoalPropertyOnEntirePath) {
			this.nodeGoalTester = null;
			this.pathGoalTester = (PathGoalTester<N>) this.graphGenerator.getGoalTester();
			;
		} else {
			this.nodeGoalTester = (NodeGoalTester<N>) this.graphGenerator.getGoalTester();
			this.pathGoalTester = null;
		}

		/* set parent discarding */
		this.config = config;
		this.parentDiscarding = config.parentDiscarding();

		/* if the node evaluator is graph dependent, communicate the generator to it */
		this.nodeEvaluator = problem.getNodeEvaluator();
		if (this.nodeEvaluator == null) {
			throw new IllegalArgumentException("Cannot work with node evaulator that is null");
		} else if (this.nodeEvaluator instanceof DecoratingNodeEvaluator<?, ?>) {
			DecoratingNodeEvaluator<N, V> castedEvaluator = (DecoratingNodeEvaluator<N, V>) this.nodeEvaluator;
			if (castedEvaluator.isGraphDependent()) {
				this.logger.info("{} is a graph dependent node evaluator. Setting its graph generator now ...", castedEvaluator);
				castedEvaluator.setGenerator(this.graphGenerator);
			}
			if (castedEvaluator.isSolutionReporter()) {
				this.logger.info("{} is a solution reporter. Register the search algo in its event bus", castedEvaluator);
				castedEvaluator.registerSolutionListener(this);
				this.solutionReportingNodeEvaluator = true;
			} else {
				this.solutionReportingNodeEvaluator = false;
			}
		} else {
			if (this.nodeEvaluator instanceof IGraphDependentNodeEvaluator) {
				this.logger.info("{} is a graph dependent node evaluator. Setting its graph generator now ...", this.nodeEvaluator);
				((IGraphDependentNodeEvaluator<N, A, V>) this.nodeEvaluator).setGenerator(this.graphGenerator);
			}

			/* if the node evaluator is a solution reporter, register in his event bus */
			if (this.nodeEvaluator instanceof ISolutionReportingNodeEvaluator) {
				this.logger.info("{} is a solution reporter. Register the search algo in its event bus", this.nodeEvaluator);
				((ISolutionReportingNodeEvaluator<N, V>) this.nodeEvaluator).registerSolutionListener(this);
				this.solutionReportingNodeEvaluator = true;
			} else {
				this.solutionReportingNodeEvaluator = false;
			}
		}

		/* add shutdown hook so as to cancel the search once the overall program is shutdown */
		Runtime.getRuntime().addShutdownHook(new Thread(() -> BestFirst.this.cancel(), "Shutdown hook thread for " + BestFirst.this));
	}

	private void labelNode(final Node<N, V> node) throws Exception {
		node.setInternalLabel(this.nodeEvaluator.f(node));
	}

	/**
	 * This method setups the graph by inserting the root nodes.
	 */
	protected void initGraph() throws Exception {
		if (!this.initialized) {
			try {
				this.nodeModelSemaphore.acquire();
				this.initialized = true;
				if (this.rootGenerator instanceof MultipleRootGenerator) {
					Collection<Node<N, V>> roots = ((MultipleRootGenerator<N>) this.rootGenerator).getRoots().stream().map(n -> this.newNode(null, n)).collect(Collectors.toList());
					for (Node<N, V> root : roots) {
						this.labelNode(root);
						this.openLock.lock();
						try {
							this.open.add(root);
						} finally {
							this.openLock.unlock();
							this.addedNodeToOpenCondition.signalAll();
						}
						root.setAnnotation("awa-level", 0);
						this.logger.info("Labeled root with {}", root.getInternalLabel());
					}
				} else {
					Node<N, V> root = this.newNode(null, ((SingleRootGenerator<N>) this.rootGenerator).getRoot());
					this.labelNode(root);
					this.openLock.lock();
					try {
						this.open.add(root);
					} finally {
						this.openLock.unlock();
						this.addedNodeToOpenCondition.signalAll();
					}
				}
			} finally {
				this.nodeModelSemaphore.release();
			}
		}
	}

	public EvaluatedSearchGraphPath<N, A, V> nextSolutionThatDominatesOpen() throws InterruptedException, AlgorithmExecutionCanceledException {
		EvaluatedSearchGraphPath<N, A, V> currentlyBestSolution = null;
		V currentlyBestScore = null;
		boolean loopCondition = true;
		while (loopCondition) {
			EvaluatedSearchGraphPath<N, A, V> solution = this.nextSolution();
			V scoreOfSolution = solution.getScore();
			if (currentlyBestScore == null || scoreOfSolution.compareTo(currentlyBestScore) < 0) {
				currentlyBestScore = scoreOfSolution;
				currentlyBestSolution = solution;
			}

			this.openLock.lock();
			try {
				loopCondition = this.open.peek().getInternalLabel().compareTo(currentlyBestScore) < 0;
			} finally {
				this.openLock.unlock();
			}
		}
		return currentlyBestSolution;
	}

	/**
	 * Find the shortest path to a goal starting from <code>start</code>.
	 *
	 * @param start
	 *            The initial node.
	 * @return A list of nodes from the initial point to a goal, <code>null</code> if a path doesn't exist.
	 */
	@Override
	public EvaluatedSearchGraphPath<N, A, V> nextSolution() throws InterruptedException, AlgorithmExecutionCanceledException {

		/* check whether solution has been canceled */
		if (this.canceled) {
			throw new IllegalStateException("Search has been canceled, no more solutions can be requested.");
		}

		/* do preliminary stuff: init graph (only for first call) and return unreturned solutions first */
		this.openLock.lock();
		try {
			this.logger.info("Starting search for next solution. Size of OPEN is {}", this.open.size());
		} finally {
			this.openLock.unlock();
		}

		try {
			this.initGraph();
		} catch (InterruptedException | AlgorithmExecutionCanceledException e) {
			throw e;
		} catch (Throwable e) {
			e.printStackTrace();
			return null;
		}
		if (!this.solutions.isEmpty()) {
			this.logger.debug("Still have solution in cache, return it.");
			EvaluatedSearchGraphPath<N, A, V> solution = this.solutions.poll();
			this.logger.info("Returning solution {} with score {}", solution.getNodes(), solution.getScore());
			return solution;
		}

		boolean loopCondition = false;
		do {
			/* Semi busy waiting on new nodes; if open queue is empty, check every time an active job finished whether there is a new node on open. */
			boolean newNodesInOpen = false;
			while (!newNodesInOpen) {
				this.openLock.lock();
				try {
					this.logger.debug("Checking new nodes: OPEN size is {} and there are {} active jobs.", this.open.size(), this.activeJobs.get());
					newNodesInOpen = !this.open.isEmpty() || this.activeJobs.get() <= 0;
				} finally {
					this.openLock.unlock();
				}

				if (!newNodesInOpen) {
					try {
						this.addedNodeToOpenCondition.await();
					} catch (InterruptedException e) {
						this.logger.info("Received interrupt signal");
						this.interrupted = true;
						throw e;
					}
				}
			}

			this.openLock.lock();
			try {
				if (this.open.isEmpty() || this.interrupted) {
					this.logger.debug("OPEN has size {} and interrupted is {}", this.open.size(), this.interrupted);
					break;
				}
			} finally {
				this.openLock.unlock();
			}

			this.openLock.lock();
			try {
				this.logger.debug("Iteration of main loop starts. Size of OPEN now {}. Now performing next expansion step.", this.open.size());
			} finally {
				this.openLock.unlock();
			}

			this.step();

			this.openLock.lock();
			try {
				if (!this.solutions.isEmpty()) {
					EvaluatedSearchGraphPath<N, A, V> solution = this.solutions.poll();
					this.logger.debug("Iteration of main loop terminated. Found a solution to return. Size of OPEN now {}", this.open.size());
					this.logger.info("Returning solution {} with score {}", solution.getNodes(), solution.getScore());
					return solution;
				}
				this.logger.debug("Iteration of main loop terminated. Size of OPEN now {}. Number of active jobs: {}", this.open.size(), this.activeJobs.get());

				loopCondition = (!this.open.isEmpty() || this.activeJobs.get() > 0) && !this.interrupted;
			} finally {
				this.openLock.unlock();
			}
		} while (loopCondition);

		if (this.interrupted) {
			this.logger.info("Algorithm was interrupted");
			throw new InterruptedException();
		}

		this.openLock.lock();
		try {
			if (this.open.isEmpty()) {
				this.logger.info("OPEN is empty, terminating (possibly returning a solution)");
			}
		} finally {
			this.openLock.unlock();
		}

		return this.solutions.isEmpty() ? null : this.solutions.poll();
	}

	/**
	 * Makes a single expansion and returns solution paths.
	 *
	 * @return The last found solution path.
	 */
	public List<NodeExpansionDescription<N, A>> nextExpansion() throws InterruptedException, AlgorithmExecutionCanceledException {
		if (!this.initialized) {
			try {
				this.initGraph();
			} catch (Throwable e) {
				e.printStackTrace();
				return null;
			}
			return this.lastExpansion;
		} else {
			this.step();
		}
		return this.lastExpansion;
	}

	protected NodeExpansionJobSubmittedEvent<N, A, V> step() throws InterruptedException, AlgorithmExecutionCanceledException {
		if (!this.beforeSelection()) {
			return null;
		}
		Node<N, V> nodeToExpand = null;
		this.openLock.lock();
		try {
			nodeToExpand = this.open.peek();
		} finally {
			this.openLock.unlock();
		}
		if (nodeToExpand == null) {
			return null;
		}
		assert this.parentDiscarding == ParentDiscarding.ALL || !this.expanded.contains(nodeToExpand.getPoint()) : "Node " + nodeToExpand.getString() + " has been selected for the second time for expansion.";
		this.afterSelection(nodeToExpand);
		return this.step(nodeToExpand);
	}

	private void checkTermination() throws InterruptedException, AlgorithmExecutionCanceledException {
		if (Thread.currentThread().isInterrupted()) {
			this.logger.debug("Received interrupt signal.");
			this.interrupted = true;
			this.shutdown();
			this.logger.info("Thread that executes the algorithm has been interrupted, emitting InterruptedException");
			throw new InterruptedException(); // if the thread itself was actively interrupted by somebody
		}
		if (this.canceled) {
			this.shutdown();
			this.logger.info("Algorithm has been canceled, emitting AlgorithmExecutionCanceledException");
			throw new AlgorithmExecutionCanceledException(); // for a controlled cancel from outside on the algorithm
		}
	}

	public NodeExpansionJobSubmittedEvent<N, A, V> step(final Node<N, V> nodeToExpand) throws InterruptedException, AlgorithmExecutionCanceledException {

		/* if search has been interrupted, do not process next step */
		this.logger.debug("Step starts. Size of OPEN now {}", this.open.size());
		this.checkTermination();
		this.lastExpansion.clear();
		assert nodeToExpand == null || !this.expanded.contains(nodeToExpand.getPoint()) : "Node selected for expansion already has been expanded: " + nodeToExpand;
		this.open.remove(nodeToExpand);
		assert !this.open.contains(nodeToExpand) : "The selected node " + nodeToExpand + " was not really removed from OPEN!";
		this.logger.debug("Removed {} from OPEN for expansion. OPEN size now {}", nodeToExpand, this.open.size());
		assert this.ext2int.containsKey(nodeToExpand.getPoint()) : "Trying to expand a node whose point is not available in the ext2int map";
		this.beforeExpansion(nodeToExpand);
		NodeExpansionJobSubmittedEvent<N, A, V> event = this.expandNode(nodeToExpand);
		this.afterExpansion(nodeToExpand);
		this.checkTermination();
		this.logger.debug("Step ends. Size of OPEN now {}", this.open.size());
		return event;
	}

	private NodeExpansionJobSubmittedEvent<N, A, V> expandNode(final Node<N, V> expandedNodeInternal) throws InterruptedException, AlgorithmExecutionCanceledException {
		if (expandedNodeInternal == null) {
			throw new IllegalArgumentException("Cannot expand node NULL");
		}
		this.graphEventBus.post(new NodeTypeSwitchEvent<Node<N, V>>(expandedNodeInternal, "or_expanding"));
		this.logger.info("Expanding node {} with f-value {}", expandedNodeInternal, expandedNodeInternal.getInternalLabel());
		assert !this.expanded.contains(expandedNodeInternal.getPoint()) : "Node " + expandedNodeInternal + " expanded twice!!";

		/* compute successors */
		this.logger.debug("Start computation of successors");
		final List<NodeExpansionDescription<N, A>> successorDescriptions = new ArrayList<>();
		this.checkTermination();

		successorDescriptions.addAll(BestFirst.this.successorGenerator.generateSuccessors(expandedNodeInternal.getPoint()));
		this.logger.debug("Finished computation of successors. Sending SuccessorComputationCompletedEvent with {} successors for {}", successorDescriptions.size(), expandedNodeInternal);
		this.graphEventBus.post(new SuccessorComputationCompletedEvent<>(expandedNodeInternal, successorDescriptions));

		/* attach successors to search graph */
		// System.out.println("Compute successors ... " + Thread.currentThread().isInterrupted());
		List<N> todoList = successorDescriptions.stream().map(d -> d.getTo()).collect(Collectors.toList());
		successorDescriptions.stream().forEach(successorDescription -> {
			if (this.interrupted || this.canceled) {
				return;
			}
			NodeBuilder nb = new NodeBuilder(todoList, expandedNodeInternal, successorDescription);
			if (this.additionalThreadsForExpansion < 1) {
				nb.run();
			} else {
				try {
					this.fComputationTickets.acquire(); // these are necessary to not flood the thread pool with queries
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
				}
				if (this.interrupted) {
					return;
				}
				if (this.interrupted || this.canceled) {
					return;
				}
				this.activeJobs.incrementAndGet();
				this.pool.submit(nb);
			}
		});
		this.checkTermination();
		this.logger.debug("Finished expansion of node {}. Size of OPEN is now {}. Number of active jobs is {}", expandedNodeInternal, this.open.size(), this.activeJobs.get());

		/* update statistics, send closed notifications, and possibly return a solution */
		this.expandedCounter++;
		this.expanded.add(expandedNodeInternal.getPoint());
		assert this.expanded.contains(expandedNodeInternal.getPoint()) : "Expanded node " + expandedNodeInternal + " was not inserted into the set of expanded nodes!";
		this.graphEventBus.post(new NodeTypeSwitchEvent<Node<N, V>>(expandedNodeInternal, "or_closed"));
		NodeExpansionJobSubmittedEvent<N, A, V> nodeCompletionEvent = new NodeExpansionJobSubmittedEvent<>(expandedNodeInternal, successorDescriptions);
		return nodeCompletionEvent;
	}

	public GraphEventBus<Node<N, V>> getEventBus() {
		return this.graphEventBus;
	}

	protected List<N> getTraversalPath(final Node<N, V> n) {
		return n.path().stream().map(p -> p.getPoint()).collect(Collectors.toList());
	}

	/**
	 * Check how many times a node was expanded.
	 *
	 * @return A counter of how many times a node was expanded.
	 */
	public int getExpandedCounter() {
		return this.expandedCounter;
	}

	public int getCreatedCounter() {
		return this.createdCounter;
	}

	public V getFValue(final N node) {
		return this.getFValue(this.ext2int.get(node));
	}

	public V getFValue(final Node<N, V> node) {
		return node.getInternalLabel();
	}

	public Map<String, Object> getNodeAnnotations(final N node) {
		Node<N, V> intNode = this.ext2int.get(node);
		return intNode.getAnnotations();
	}

	public Object getNodeAnnotation(final N node, final String annotation) {
		Node<N, V> intNode = this.ext2int.get(node);
		return intNode.getAnnotation(annotation);
	}

	@Override
	public void cancel() {
		this.logger.info("Set cancel flag to true.");
		this.canceled = true;
		this.shutdown();
	}

	private void shutdown() {
		this.logger.info("Invoking shutdown routine ...");
		this.state = AlgorithmState.inactive;
		if (this.pool != null) {
			this.logger.info("Triggering shutdown of builder thread pool with interrupt");
			this.pool.shutdownNow();
		}
		if (this.nodeEvaluator instanceof ICancelableNodeEvaluator) {
			this.logger.info("Canceling node evaluator.");
			((ICancelableNodeEvaluator) this.nodeEvaluator).cancel();
		}
		if (this.timeoutSubmitter != null) {
			this.timeoutSubmitter.close();
		}
		if (this.timeoutTimer != null) {
			this.logger.info("Canceling timeout.");
			this.timeoutTimer.cancel();
		}
	}

	public boolean isInterrupted() {
		return this.interrupted;
	}

	public List<N> getCurrentPathToNode(final N node) {
		return this.ext2int.get(node).externalPath();
	}

	public Node<N, V> getInternalRepresentationOf(final N node) {
		return this.ext2int.get(node);
	}

	public List<Node<N, V>> getOpenSnapshot() {
		return Collections.unmodifiableList(new ArrayList<>(this.open));
	}

	protected synchronized Node<N, V> newNode(final Node<N, V> parent, final N t2) {
		return this.newNode(parent, t2, null);
	}

	public INodeEvaluator<N, V> getNodeEvaluator() {
		return this.nodeEvaluator;
	}

	protected synchronized Node<N, V> newNode(final Node<N, V> parent, final N t2, final V evaluation) {
		assert !this.open.contains(parent) : "Parent node " + parent + " is still on OPEN, which must not be the case!";

		/* create new node and check whether it is a goal */
		Node<N, V> newNode = new Node<>(parent, t2);
		if (evaluation != null) {
			newNode.setInternalLabel(evaluation);
		}

		/* check loop */
		assert parent == null || !parent.externalPath().contains(t2) : "There is a loop in the underlying graph. The following path contains the last node twice: "
				+ newNode.externalPath().stream().map(n -> n.toString()).reduce("", (s, t) -> s + "\n\t\t" + t);

		/* currently, we only support tree search */
		assert !this.ext2int.containsKey(t2) : "Reached node " + t2 + " for the second time.\nt\tFirst path:" + this.ext2int.get(t2).externalPath().stream().map(n -> n.toString()).reduce("", (s, t) -> s + "\n\t\t" + t) + "\n\tSecond Path:"
				+ newNode.externalPath().stream().map(n -> n.toString()).reduce("", (s, t) -> s + "\n\t\t" + t);

		/* register node in map and create annotation object */
		this.ext2int.put(t2, newNode);

		/* detect whether node is solution */
		if (this.checkGoalPropertyOnEntirePath ? this.pathGoalTester.isGoal(newNode.externalPath()) : this.nodeGoalTester.isGoal(newNode.getPoint())) {
			newNode.setGoal(true);
		}

		/* send events for this new node */
		if (parent == null) {
			this.graphEventBus.post(new GraphInitializedEvent<Node<N, V>>(newNode));
		} else {
			this.graphEventBus.post(new NodeReachedEvent<Node<N, V>>(parent, newNode, "or_" + (newNode.isGoal() ? "solution" : "created")));
			this.logger.debug("Sent message for creation of node {} as a successor of {}", newNode, parent);
		}
		return newNode;
	}

	/**
	 * This method can be used to create an initial graph different from just root nodes. This can be interesting if the search is distributed and we want to search only an excerpt of the original one.
	 *
	 * @param initialNodes
	 */
	public void bootstrap(final Collection<Node<N, V>> initialNodes) {

		if (this.initialized) {
			throw new UnsupportedOperationException("Bootstrapping is only supported if the search has already been initialized.");
		}

		/* now initialize the graph */
		try {
			this.initGraph();
		} catch (Throwable e) {
			e.printStackTrace();
			return;
		}

		/* remove previous roots from open */
		this.open.clear();

		/* now insert new nodes, and the leaf ones in open */
		for (Node<N, V> node : initialNodes) {
			this.insertNodeIntoLocalGraph(node);
			this.open.add(this.getLocalVersionOfNode(node));
		}
	}

	protected void insertNodeIntoLocalGraph(final Node<N, V> node) {
		Node<N, V> localVersionOfParent = null;
		List<Node<N, V>> path = node.path();
		Node<N, V> leaf = path.get(path.size() - 1);
		for (Node<N, V> nodeOnPath : path) {
			if (!this.ext2int.containsKey(nodeOnPath.getPoint())) {
				assert nodeOnPath.getParent() != null : "Want to insert a new node that has no parent. That must not be the case! Affected node is: " + nodeOnPath.getPoint();
				assert this.ext2int.containsKey(nodeOnPath.getParent().getPoint()) : "Want to insert a node whose parent is unknown locally";
				Node<N, V> newNode = this.newNode(localVersionOfParent, nodeOnPath.getPoint(), nodeOnPath.getInternalLabel());
				if (!newNode.isGoal() && !newNode.getPoint().equals(leaf.getPoint())) {
					this.getEventBus().post(new NodeTypeSwitchEvent<Node<N, V>>(newNode, "or_closed"));
				}
				localVersionOfParent = newNode;
			} else {
				localVersionOfParent = this.getLocalVersionOfNode(nodeOnPath);
			}
		}
	}

	@Override
	public boolean hasNext() {
		return this.state != AlgorithmState.inactive;
	}

	@Override
	public AlgorithmEvent next() {
		try {
			switch (this.state) {
			case created: {
				this.logger.info("Initializing BestFirst search {} with {} CPUs and a timeout of {}ms", this, this.config.cpus(), this.config.timeout());
				this.timeoutTimer = new Timer("Timeouter-" + this);
				this.timeoutTimer.schedule(new TimerTask() {

					@Override
					public void run() {
						BestFirst.this.logger.info("Invoking cancel on BestFirst search {}", BestFirst.this);
						BestFirst.this.cancel();
					}
				}, this.config.timeout());
				this.parallelizeNodeExpansion(this.config.cpus());
				this.initGraph();
				this.state = AlgorithmState.active;
				AlgorithmEvent event = new AlgorithmInitializedEvent();
				this.graphEventBus.post(event);
				return event;
			}
			case active: {
				if (!this.pendingSolutionFoundEvents.isEmpty()) {
					return this.pendingSolutionFoundEvents.poll(); // these already have been posted over the event bus but are now returned to the controller for respective handling
				}
				AlgorithmEvent event;
				try {
					event = this.step();
					if (event == null) {
						event = new AlgorithmFinishedEvent();
						this.state = AlgorithmState.inactive;
					}
				} catch (AlgorithmExecutionCanceledException e) {
					event = new AlgorithmFinishedEvent();
				}
				this.graphEventBus.post(event);
				return event;
			}
			default:
				throw new IllegalStateException("BestFirst search is in state " + this.state + " in which next must not be called!");
			}
		} catch (Throwable e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * This is relevant if we work with several copies of a node (usually if we need to copy the search space somewhere).
	 *
	 * @param node
	 * @return
	 */
	protected Node<N, V> getLocalVersionOfNode(final Node<N, V> node) {
		return this.ext2int.get(node.getPoint());
	}

	/* hooks */
	protected void afterInitialization() {
	}

	protected boolean beforeSelection() {
		return true;
	}

	protected void afterSelection(final Node<N, V> node) {
	}

	protected void beforeExpansion(final Node<N, V> node) {
	}

	protected void afterExpansion(final Node<N, V> node) {
	}

	@Override
	public Iterator<AlgorithmEvent> iterator() {
		return this;
	}

	private void registerSolutionCandidateViaEvent(final GraphSearchSolutionCandidateFoundEvent<N, A, V> solutionEvent) {
		EvaluatedSearchGraphPath<N, A, V> solution = solutionEvent.getSolutionCandidate();
		this.solutions.add(solutionEvent.getSolutionCandidate());
		this.pendingSolutionFoundEvents.add(solutionEvent);
		this.graphEventBus.post(solutionEvent);
		if (this.bestSeenSolution == null || solution.getScore().compareTo(this.bestSeenSolution.getScore()) < 0) {
			this.bestSeenSolution = solution;
		}
	}

	@Subscribe
	public void receiveSolutionCandidateEvent(final GraphSearchSolutionCandidateFoundEvent<N, A, V> solutionEvent) {
		try {
			this.logger.info("Received solution with f-value {} and annotations {}", solutionEvent.getSolutionCandidate().getScore(), solutionEvent.getSolutionCandidate().getAnnotations());
			this.registerSolutionCandidateViaEvent(solutionEvent);
		} catch (Throwable e) {
			e.printStackTrace();
		}
	}

	@Subscribe
	public void receiveSolutionCandidateAnnotationEvent(final SolutionAnnotationEvent<N, A, V> event) {
		try {
			this.logger.debug("Received solution annotation: {}", event);
			this.graphEventBus.post(event);
		} catch (Throwable e) {
			e.printStackTrace();
		}
	}

	@Subscribe
	public void receiveNodeAnnotationEvent(final NodeAnnotationEvent<N> event) {
		try {
			N nodeExt = event.getNode();
			this.logger.debug("Received annotation {} with value {} for node {}", event.getAnnotationName(), event.getAnnotationValue(), event.getNode());
			if (!this.ext2int.containsKey(nodeExt)) {
				throw new IllegalArgumentException("Received annotation for a node I don't know!");
			}
			Node<N, V> nodeInt = this.ext2int.get(nodeExt);
			nodeInt.setAnnotation(event.getAnnotationName(), event.getAnnotationValue());
		} catch (Throwable e) {
			e.printStackTrace();
		}
	}

	public int getAdditionalThreadsForExpansion() {
		return this.additionalThreadsForExpansion;
	}

	private void parallelizeNodeExpansion(final int threadsForExpansion) {
		if (this.pool != null) {
			throw new UnsupportedOperationException("The number of additional threads can be only set once per search!");
		}
		if (threadsForExpansion < 1) {
			throw new IllegalArgumentException("Number of threads should be at least 1 for " + this.getClass().getName());
		}
		this.fComputationTickets = new Semaphore(threadsForExpansion);
		this.additionalThreadsForExpansion = threadsForExpansion;
		AtomicInteger counter = new AtomicInteger(0);
		this.pool = Executors.newFixedThreadPool(threadsForExpansion, r -> {
			Thread t = new Thread(r);
			t.setName("ORGraphSearch-worker-" + counter.incrementAndGet());
			return t;
		});
	}

	public int getTimeoutForComputationOfF() {
		return this.timeoutForComputationOfF;
	}

	public void setTimeoutForComputationOfF(final int timeoutInMS, final INodeEvaluator<N, V> timeoutEvaluator) {
		this.timeoutForComputationOfF = timeoutInMS;
		this.timeoutNodeEvaluator = timeoutEvaluator;
	}

	/**
	 * @return the openCollection
	 */
	public OpenCollection<Node<N, V>> getOpen() {
		return this.open;
	}

	/**
	 * @param open
	 *            the openCollection to set
	 */
	public void setOpen(final OpenCollection<Node<N, V>> collection) {

		collection.clear();
		collection.addAll(this.open);
		this.open = collection;
	}

	@Override
	public void setLoggerName(final String name) {
		this.logger.info("Switching logger from {} to {}", this.logger.getName(), name);
		this.logger = LoggerFactory.getLogger(name);
		this.logger.info("Activated logger {} with name {}", name, this.logger.getName());
	}

	@Override
	public String getLoggerName() {
		return this.logger.getName();
	}

	public Queue<EvaluatedSearchGraphPath<N, A, V>> getSolutionQueue() {
		return this.solutions;
	}

	@Override
	public EvaluatedSearchGraphPath<N, A, V> call() throws InterruptedException, AlgorithmExecutionCanceledException {
		this.logger.info("Invoking \"call\" on BestFirst");
		try {
			while (this.state != AlgorithmState.inactive) {
				this.next();
			}
		} catch (RuntimeException e) {
			if (e.getCause() instanceof InterruptedException) {
				throw (InterruptedException) e.getCause();
			} else if (e.getCause() instanceof AlgorithmExecutionCanceledException) {
				throw (AlgorithmExecutionCanceledException) e.getCause();
			} else {
				throw e;
			}
		}
		return this.bestSeenSolution;
	}

	@Override
	public void setNumCPUs(final int numberOfCPUs) {
		this.config.setProperty(IAlgorithmConfig.K_CPUS, String.valueOf(numberOfCPUs));
	}

	@Override
	public void registerListener(final Object listener) {
		this.graphEventBus.register(listener);
	}

	@Override
	public int getNumCPUs() {
		return this.config.cpus();
	}

	@Override
	public void setTimeout(final int timeout, final TimeUnit timeUnit) {
		if (timeUnit != TimeUnit.MILLISECONDS) {
			throw new IllegalArgumentException("Currently only support for ms");
		}
		this.config.setProperty(IAlgorithmConfig.K_TIMEOUT, String.valueOf(timeout));
	}

	@Override
	public int getTimeout() {
		return this.config.timeout();
	}

	@Override
	public TimeUnit getTimeoutUnit() {
		return TimeUnit.MILLISECONDS;
	}
}
