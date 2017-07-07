package jaicore.search.algorithms.parallel.parallelevaluation.local.core;

import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;

import jaicore.search.algorithms.standard.core.NodeEvaluator;
import jaicore.search.algorithms.standard.core.ORGraphSearch;
import jaicore.search.structure.core.GraphGenerator;
import jaicore.search.structure.core.Node;
import jaicore.search.structure.events.NodeTypeSwitchEvent;

public class ParallelizedORGraphSearch<T, A, V extends Comparable<V>> extends ORGraphSearch<T, A, V> {

	private final int NUM_THREADS = 4;
	private final Semaphore fComputationTickets = new Semaphore(NUM_THREADS);
	private final ExecutorService pool = Executors.newFixedThreadPool(NUM_THREADS);
	private final AtomicInteger activeJobs = new AtomicInteger(0);
	private final Timer timeouter = new Timer();
	private final ITimeoutNodeEvaluator<T, V> timeoutNodeEvaluator;
	private final int timeout;

	public ParallelizedORGraphSearch(GraphGenerator<T, A> graphGenerator, NodeEvaluator<T, V> pNodeEvaluator, int timeout) {
		this(graphGenerator, pNodeEvaluator, n -> null, timeout);
	}

	public ParallelizedORGraphSearch(GraphGenerator<T, A> graphGenerator, NodeEvaluator<T, V> pNodeEvaluator, ITimeoutNodeEvaluator<T, V> timeoutNodeEvaluator, int timeout) {
		super(graphGenerator, pNodeEvaluator);
		this.timeoutNodeEvaluator = timeoutNodeEvaluator;
		this.timeout = timeout;
	}

	protected boolean terminates() {
		if (activeJobs.get() > 0)
			return false;
		return super.terminates();
	}

	protected boolean beforeInsertionIntoOpen(Node<T, V> node) {

		try {
			fComputationTickets.acquire();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		activeJobs.incrementAndGet();
		final Future<?> job = pool.submit(new Runnable() {

			@Override
			public void run() {
				
				/* compute f-value for node; possibly use the escape value from the timeout node evaluator */
				V label = nodeEvaluator.f(node);
				if (Thread.interrupted()) {
					eventBus.post(new NodeTypeSwitchEvent<>(node, "or_timedout"));
					if (label != null)
						System.err.println("A timeout was observed but the label was still computed. It is likely that the logic that computes f does not observe interrupts!");
					label = timeoutNodeEvaluator.f(node);
				}
				
				/* only if we have a non-null label, update the node label and insert the node into open */
				if (label != null) {
					node.setInternalLabel(label);
					open.add(node);
				}
				
				/* in any case, free the resources to compute f-values */
				activeJobs.decrementAndGet();
				fComputationTickets.release();
			}
		});
		
		/* set timeout for the job */
		TimerTask t = new TimerTask() {

			@Override
			public void run() {
				if (!job.isDone()) {
					job.cancel(true);
				}
			}
		};
		timeouter.schedule(t, timeout);
		return false;
	}
}
