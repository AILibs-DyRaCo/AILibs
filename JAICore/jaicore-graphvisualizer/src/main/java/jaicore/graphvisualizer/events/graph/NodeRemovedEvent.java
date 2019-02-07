package jaicore.graphvisualizer.events.graph;

public class NodeRemovedEvent<T> implements GraphEvent {

	private final T node;
	public final String name = "NodeRemovedEvent"; // changed name to NodeRemovedEvent. before it was: "§NodeRemovedEvent";

	public NodeRemovedEvent(T node) {
		super();
		this.node = node;
	}

	public T getNode() {
		return node;
	}

}
