package jaicore.search.algorithms.standard.nqueens;

import java.util.ArrayList;
import java.util.List;

public class QueenNode {
	/*
	 * Helperclass to store the positions of the queen.
	 */
	public class Position {
		int x;
		int y;
		
		public Position(int x, int y) {
			this.x= x;
			this.y = y;
		}
		
		public Position(Position pos) {
			this.x = pos.getX();
			this.y = pos.getY();
		}

		public int getX() {
			return x;
		}


		public int getY() {
			return y;
		}

		public boolean equals(Object obj) {
			Position p =(Position) obj;
			if(p.getX() == x && p.getY() == y)
				return true;
			else
				return false;
		}
		
		@Override
		public String toString() {
			return "("+x+"," + y+ ")";
		}

		public boolean attack(int i, int j, int dimension ) {
			if(i == x || j == y || isOnDiag(i,j, dimension))
				return true;
			return false;
		}

		private boolean isOnDiag(int i, int j, int dimension) {
			int ex = x;
			int ey = y;
			//left up
			while(ex >= 0 && ey >= 0) {
				ex --;
				ey --;
				if(ex == i && ey == j) 
					return true;				
			}
			//right up
			ex = x;
			ey = y;
			while(ex >= 0 && ey < dimension) {
				ex --;
				ey ++;
				if(ex == i && ey == j) 
					return true;				
			}
			//right down
			ex = x;
			ey = y;
			while(ex <dimension  && ey < dimension) {
				ex ++;
				ey ++;
				if(ex == i && ey == j) 
					return true;
			}
			//left down
			ex = x;
			ey = y;
			while(ex < dimension && ey >= 0) {
				ex ++;
				ey --;
				if(ex == i && ey == j) 
					return true;				
			}
			return false;
		}
		
		
	}

	List<Position> positions;
	
	int dimension;
	
	/**
	 * Creates a QueenNode with a empty board
	 * @param dimension
	 * 		The dimension of the board.
	 */
	public QueenNode(int dimension) {
		this.positions = new ArrayList<>();
		this.dimension = dimension;
	}
	
	/**
	 * Creates a QueenNode with one Queen on it
	 * @param x
	 * 	The row position of the queen.
	 * @param y
	 * 	The column position of the queen.
	 * @param dimension
	 * 	The dimension of the board.
	 */
	public QueenNode(int x, int y, int dimension) {
		positions = new ArrayList<>();
		positions.add(new Position(x, y));
		this.dimension = dimension;
		
	}
	
	/**
	 * Creates a QueenNode with exiting positions of other queens
	 * @param pos
	 * 		The  positions of the other queens.
	 * @param x
	 * 		The row position of the newly placed queen.
	 * @param y
	 * 		The column position of the newly placed queen.
	 * @param dimension
	 * 		The dimension of the board.
	 */
	public QueenNode(List<Position> pos, int x, int y, int dimension) {
		for(Position p:pos)
			this.positions.add(new Position(p));
		this.positions = pos;
		positions.add(new Position(x, y));
		this.dimension = dimension;
	}
	
	/**
	 * Creates a new QueenNode out of another QueenNode to add a new queen.
	 * @param n
	 * 		The old QueenNode.
	 * @param x
	 * 		The row position of the new queen.
	 * @param y
	 * 		The column position of the new queen.
	 */
	public QueenNode(QueenNode n, int x, int y) {
		//Cloning the list
		this.positions = new ArrayList<>(n.getPositions().size());
		for(Position p : n.getPositions())
			this.positions.add(new Position(p));
		
		positions.add(new Position(x, y));
		this.dimension = n.getDimension();
	}
	
	


	public List<Position> getPositions(){
		return this.positions;
	}
	
	public int getDimension() {
		return this.dimension;
	}
	
	@Override
	public String toString() {
		String s = "";
		for(int i = 0; i< dimension; i++) 
			s+="----";
		
		s+="\n|";
		
		for(int i = 0; i < dimension; i++) {
			for(int j = 0; j < dimension; j++) {
				if(positions.contains(new Position(i,j)))
					s+= " Q |";
				else
					s+= "   |";
			}
			s+= "\n";
			for(int j = 0; j<dimension; j++) {
				s+= "----";
			}
			if(i < dimension-1)
				s+= "\n|";
		}
		
		return s;
	}
	
	/**
	 * Checks if a cell is attacked by the queens on the board
	 * @param i
	 * 		The row of the cell to be checked.
	 * @param j
	 * 		The collumn of the cell to be checked.
	 * @return
	 * 		<code>true</code> if the cell is attacked, <code>false</code> otherwise.
	 */
	public boolean attack(int i, int j) {
		for(Position p: positions)
			if(p.attack(i, j, dimension))
				return true;
	
		return false;
	}
	public String toStringAttack() {
		String s = "";
		for(int i = 0; i< dimension; i++) 
			s+="----";
		
		s+="\n|";
		
		for(int i = 0; i < dimension; i++) {
			for(int j = 0; j < dimension; j++) {
				if(positions.contains(new Position(i,j)))
					s+= " Q |";
				
				else {
					boolean attack = false;
					for(Position p : positions) {
						attack = attack(i,j);
						if(attack)
							break;
					}
					if(attack)
						s+= " O |";
					else
						s+= "   |";
				}
			}
			s+= "\n";
			for(int j = 0; j<dimension; j++) {
				s+= "----";
			}
			if(i < dimension-1)
				s+= "\n|";
		}
		
		return s;
	}
	
	public int getNumberOfQueens() {
		return positions.size();
	}
	
	/**
	 * Returns the number of attacked cells of the current boardconfiguration
	 * @return	
	 * 		The number of attacked cells.
	 */
	public int getNumberOfAttackedCells() {
		int attackedCells = positions.size() * dimension;
		for(int i = positions.size(); i < dimension; i++) {
			for(int j  =0 ; j < dimension; j++) {
				if(this.attack(i, j))
					attackedCells ++;
			}
		}
		return attackedCells;
	}
	
	/**
	 * Returns the number of attacked cells in the next free row from top down.
	 * @return
	 * 		The number of attacked cells in the next row.
	 */
	public int getNumberOfAttackedCellsInNextRow() {
		int attackedCells = 0;
		for(int i = 0; i < dimension; i++) {
			if(this.attack(dimension, i))
				attackedCells ++;
		}
		return attackedCells;
	}
}
