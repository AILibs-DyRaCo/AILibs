package jaicore.search.algorithms.interfaces;

import java.util.List;

import jaicore.search.structure.core.Node;

public interface IORGraphSearch<T, A, V extends Comparable<V>> {

	public List<T> nextSolution();

	public V getFValue(T node);

	public V getFValue(Node<T, V> node);

	public V getFOfReturnedSolution(List<T> solution);

	public void cancel();

	public Node<T, V> getInternalRepresentationOf(T node);

	public List<Node<T, V>> getOpenSnapshot();
}
