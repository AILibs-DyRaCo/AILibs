package jaicore.search.algorithms.standard.npuzzle;

import java.util.Set;

import jaicore.search.structure.graphgenerator.NodeGoalTester;

public class NPuzzleStarGenerator extends NPuzzleGenerator {
	
	Set<String> needed;

	public NPuzzleStarGenerator(int dim, int shuffle) {
		super(dim, shuffle);
	}
	
	
	@Override
	public NodeGoalTester<NPuzzleNode> getGoalTester() {
		return n->{
			if(needed.isEmpty()){
				int[][] board= n.getBoard();
				if(board[dimension-1][dimension-1]!= 0)
					return false;
				else {
					int sol =1;
					for(int i= 0; i < dimension; i++) 
						for(int j = 0; j < dimension; j++){
							if(i != dimension -1 & j != dimension -1)
								if(board[i][j] != sol)
									return false;
							
							sol ++;
						}
					
					return true;
				}
			}
			else
				return false;
		};
	}
	
	
	
	@Override
	public NPuzzleNode move(NPuzzleNode n, String m) {
		if(needed.contains(m))
			needed.remove(m);
		return super.move(n, m);

	}
	
	

}
