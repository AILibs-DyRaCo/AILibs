package util.planning.model.task.rtn;

import java.util.Collection;

import util.planning.model.ceoc.CEOCOperation;

public class RTNPlanningDomain {

	private final Collection<CEOCOperation> operations;
	private final Collection<RTNMethod> methods;

	public RTNPlanningDomain(Collection<CEOCOperation> operations, Collection<RTNMethod> methods) {
		super();
		this.operations = operations;
		this.methods = methods;
	}

	public Collection<CEOCOperation> getOperations() {
		return operations;
	}

	public Collection<RTNMethod> getMethods() {
		return methods;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((methods == null) ? 0 : methods.hashCode());
		result = prime * result + ((operations == null) ? 0 : operations.hashCode());
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
		RTNPlanningDomain other = (RTNPlanningDomain) obj;
		if (methods == null) {
			if (other.methods != null)
				return false;
		} else if (!methods.equals(other.methods))
			return false;
		if (operations == null) {
			if (other.operations != null)
				return false;
		} else if (!operations.equals(other.operations))
			return false;
		return true;
	}
}
