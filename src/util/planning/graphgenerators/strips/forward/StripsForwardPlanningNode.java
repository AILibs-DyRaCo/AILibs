package util.planning.graphgenerators.strips.forward;

import util.logic.Monom;
import util.planning.model.core.Action;

public class StripsForwardPlanningNode {

	private final Monom state;
	private final Action actionToReachState;

	public StripsForwardPlanningNode(Monom state, Action actionToReachState) {
		super();
		this.state = state;
		this.actionToReachState = actionToReachState;
	}

	public Monom getState() {
		return state;
	}

	public Action getActionToReachState() {
		return actionToReachState;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((state == null) ? 0 : state.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		StripsForwardPlanningNode other = (StripsForwardPlanningNode) obj;
		if (state == null) {
			if (other.state != null)
				return false;
		} else if (!state.equals(other.state))
			return false;
		return true;
	}
}
