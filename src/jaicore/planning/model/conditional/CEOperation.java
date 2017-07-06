package jaicore.planning.model.conditional;

import java.util.List;
import java.util.Map;

import jaicore.logic.CNFFormula;
import jaicore.logic.Monom;
import jaicore.logic.VariableParam;
import jaicore.planning.model.core.Operation;

public class CEOperation extends Operation {
	
	private final Map<CNFFormula,Monom> addLists, deleteLists;

	public CEOperation(String name, List<VariableParam> params, Monom precondition, Map<CNFFormula,Monom> addLists, Map<CNFFormula,Monom> deleteLists) {
		super(name, params, precondition);
		this.addLists = addLists;
		this.deleteLists = deleteLists;
	}

	public Map<CNFFormula, Monom> getAddLists() {
		return addLists;
	}

	public Map<CNFFormula, Monom> getDeleteLists() {
		return deleteLists;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = super.hashCode();
		result = prime * result + ((addLists == null) ? 0 : addLists.hashCode());
		result = prime * result + ((deleteLists == null) ? 0 : deleteLists.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (!super.equals(obj))
			return false;
		if (getClass() != obj.getClass())
			return false;
		CEOperation other = (CEOperation) obj;
		if (addLists == null) {
			if (other.addLists != null)
				return false;
		} else if (!addLists.equals(other.addLists))
			return false;
		if (deleteLists == null) {
			if (other.deleteLists != null)
				return false;
		} else if (!deleteLists.equals(other.deleteLists))
			return false;
		return true;
	}

	
}
