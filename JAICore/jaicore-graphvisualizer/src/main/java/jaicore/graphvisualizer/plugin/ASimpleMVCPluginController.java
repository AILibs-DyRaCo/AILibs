package jaicore.graphvisualizer.plugin;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

import jaicore.basic.algorithm.events.AlgorithmEvent;
import jaicore.graphvisualizer.events.graph.bus.HandleAlgorithmEventException;
import jaicore.graphvisualizer.events.gui.GUIEvent;
import jaicore.graphvisualizer.plugin.controlbar.ResetEvent;
import jaicore.graphvisualizer.plugin.timeslider.GoToTimeStepEvent;

public abstract class ASimpleMVCPluginController<M extends ASimpleMVCPluginModel<?, ?>, V extends ASimpleMVCPluginView<?, ?, ?>> extends Thread implements IGUIPluginController {

	private final Queue<AlgorithmEvent> eventQueue;

	private final V view;
	private final M model;

	public ASimpleMVCPluginController(M model, V view) {
		super();
		this.model = model;
		this.view = view;
		eventQueue = new ConcurrentLinkedQueue<>();
	}

	public M getModel() {
		return model;
	}

	public V getView() {
		return view;
	}

	@Override
	public final void handleAlgorithmEvent(AlgorithmEvent algorithmEvent) throws HandleAlgorithmEventException {
		eventQueue.add(algorithmEvent);
	}

	@Override
	public void run() {
		while (true) {
			AlgorithmEvent event = eventQueue.poll();
			if (event != null) {
				handleAlgorithmEventInternally(event);
			}
		}
	}

	protected abstract void handleAlgorithmEventInternally(AlgorithmEvent algorithmEvent);
	
	@Override
	public void handleGUIEvent(GUIEvent guiEvent) {
		if (guiEvent instanceof ResetEvent || guiEvent instanceof GoToTimeStepEvent) {
			getModel().clear();
			getView().clear();
		}
	}
}
