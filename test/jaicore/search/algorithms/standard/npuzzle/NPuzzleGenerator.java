package jaicore.search.algorithms.standard.npuzzle;

import java.util.ArrayList;
import java.util.List;

import jaicore.search.structure.core.GraphGenerator;
import jaicore.search.structure.core.NodeExpansionDescription;
import jaicore.search.structure.core.NodeType;
import jaicore.search.structure.graphgenerator.NodeGoalTester;
import jaicore.search.structure.graphgenerator.RootGenerator;
import jaicore.search.structure.graphgenerator.SingleRootGenerator;
import jaicore.search.structure.graphgenerator.SuccessorGenerator;

/**
 * A simple generator for the normal NPuzzleProblem
 * @author jkoepe
 *
 */
public class NPuzzleGenerator implements GraphGenerator<NPuzzleNode, String>{
	
	int dimension;
	SingleRootGenerator<NPuzzleNode> root;
	
	public NPuzzleGenerator(int dim) {
		this.dimension = dim;
		root = ()-> new NPuzzleNode(dim);
//		root = ()-> new NPuzleNode(dim,100);
	}

	@Override
	public RootGenerator<NPuzzleNode> getRootGenerator() {
		return root;
	}

	@Override
	public SuccessorGenerator<NPuzzleNode, String> getSuccessorGenerator() {
		return n -> {
			List<NodeExpansionDescription<NPuzzleNode, String>> successors = new ArrayList<>();
			
			//Possible successors
			if(n.getEmptyX()> 0)//move left
				successors.add(new NodeExpansionDescription<NPuzzleNode, String>(n,move(n, "l"), "l", NodeType.OR));
			
			if(n.getEmptyX()< dimension-1)//move right
				successors.add(new NodeExpansionDescription<NPuzzleNode, String>(n,move(n, "r"), "r", NodeType.OR));
			
			if(n.getEmptyY()>0)//move up
				successors.add(new NodeExpansionDescription<NPuzzleNode, String>(n,move(n, "u"), "u", NodeType.OR));
			
			if(n.getEmptyY()< dimension -1)//move down
				successors.add(new NodeExpansionDescription<NPuzzleNode, String>(n,move(n, "d"), "d", NodeType.OR));
			
			return successors;
		};
	}

	@Override
	public NodeGoalTester<NPuzzleNode> getGoalTester() {
		return n->{
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
		};
	}

	@Override
	public boolean isSelfContained() {
		
		return false;
	}
	
	
	public NPuzzleNode move(NPuzzleNode n, String move) {
//		if(move.equals("l"))
//			return move(n, 0,-1);
//		
//		if(move.equals("r"))
//			return move(n, 0,1);
//		
//		if(move.equals("u"))
//			return move(n, 1,0);
//		
//		if(move.equals("d"))
//			return move(n, -1,0);
//		
//		System.out.println("No Valid move. No Move is executed");
//		return null;
		switch(move) {
			case "l" : 
				return move(n, 0,-1);
			case "r" : 
				return move(n, 0, 1);
			case "d" : 
				return move(n, 1, 0);
			case "u" : 
				return move(n, -1, 0);
			default:
				System.out.println("No Valid move.");
				return null;
		}
	}
	
	public NPuzzleNode move(NPuzzleNode n,int y, int x) {
		//cloning the board for the new node
		
		if(x == y || Math.abs(x)>1 || Math.abs(y)>1) {
			System.out.println("No valid move. No move is executed");
			return null;
		}
		
		int[][] b = new int[dimension][dimension];
		for(int i = 0; i< dimension; i++) {
			for(int j= 0; j < dimension ; j++) {
				b[i][j] = n.getBoard()[i][j];
			}
		}
		int eX = n.getEmptyX();
		int eY = n.getEmptyY();
//		int help = b[eY][eX];
		b[eY][eX] = b[eY +y][eX+x];
		b[eY+y][eX+x] = 0;
		
		NPuzzleNode node = new NPuzzleNode(b, eX+x, eY+y);
		
		return node;		
	}
	
	
	
}
