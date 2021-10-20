package tensors.Double;

import java.io.Serializable;

import java.util.Arrays;
import java.util.Random;

public class Matrix implements Serializable {
	
	private static final long serialVersionUID = 1L;
	
	private int rows, cols;
	private double[][] data;
	
	private static final Random rand = new Random();
	
	public Matrix(int rows, int cols) {
		
		this.rows = rows;
		this.cols = cols;
		
		data = new double[getRows()][getCols()];
		//setAll(0.0f);
	}
	
	public Matrix(double[] inputData, int rows, int cols) {
		
		this.rows = rows;
		this.cols = cols;
		
		data = new double[getRows()][getCols()];
		
		if (rows * cols != inputData.length)
			throw new RuntimeException("The product rows * cols must be the same as the input data length");
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				data[i][j] = inputData[i * getCols() + j];
	}
	
	public static Matrix identity(int n) {
		
		double[][] resultData = new double[n][n];
		for (int i = 0; i < n; i++)
			resultData[i][i] = 1.0f;
		
		return new Matrix(resultData);
	}
	
	public static double[][] dataCopy(double[][] prevData){
		
		double[][] result = new double[prevData.length][prevData[0].length];
		for (int i = 0; i < prevData.length; i++)
			for (int j = 0; j < prevData[0].length; j++)
				result[i][j] = prevData[i][j];
		
		return result;
	}
	
	public Matrix randomize(double deviation) {
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				set(i, j, (double) rand.nextGaussian() * deviation);
		return this;
	}
	
	public Matrix randomize(double min, double max) {
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				set(i, j, (double) rand.nextGaussian() * (max - min) + min);
		return this;
	}
	
	public Matrix randomizeInt(int max) {
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				set(i, j, rand.nextInt(max));
		return this;
	}
	
	public Matrix add(Matrix mat) {
		
		if (getRows() != mat.getRows() || getCols() != mat.getCols())
			throw new RuntimeException("Matrix dimensions mismatch in dynamic method: Matrix add(Matrix mat): ("	
					+ getRows() + "," + getCols() + ") + (" + mat.getRows() + "," + mat.getCols() + ") dimensions must be the same");
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				data[i][j] += mat.get(i, j);	
		return this;
	}
	
	public static Matrix add(Matrix mat1, Matrix mat2) {
		
		if (mat1.getRows() != mat2.getRows() || mat1.getCols() != mat2.getCols())
			throw new RuntimeException("Matrix dimensions mismatch in static method: Matrix add(Matrix mat1, Matrix mat2): ("
					+ mat1.getRows() + "," + mat1.getCols() + ") + (" + mat2.getRows() + "," + mat2.getCols() + ") dimensions must be the same");
			
		double[][] resultData = new double[mat1.getRows()][mat1.getCols()];
		for (int i = 0; i < mat1.getRows(); i++)
			for (int j = 0; j < mat1.getCols(); j++)
				resultData[i][j] = mat1.get(i, j) + mat2.get(i, j);
		
		return new Matrix(resultData);
	}
	
	public Matrix add(double scalar) {
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				data[i][j] += scalar;
		return this;
	}
	
	public static Matrix add(Matrix mat, double scalar) {
		
		double[][] resultData = new double[mat.getRows()][mat.getCols()];
		for (int i = 0; i < mat.getRows(); i++)
			for (int j = 0; j < mat.getCols(); j++)
				resultData[i][j] = mat.get(i, j) + scalar;
		
		return new Matrix(resultData);
	}
	
	public Matrix sub(Matrix mat) {
		
		if (getRows() != mat.getRows() || getCols() != mat.getCols())
			throw new RuntimeException("Matrix dimensions mismatch in dynamic method: Matrix sub(Matrix mat): ("
					+ getRows() + "," + getCols() + ") - (" + mat.getRows() + "," + mat.getCols() + ") dimensions must be the same");
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				data[i][j] -= mat.get(i, j);
		return this;
	}
	
	public static Matrix sub(Matrix mat1, Matrix mat2) {
		
		if (mat1.getRows() != mat2.getRows() || mat1.getCols() != mat2.getCols())
			throw new RuntimeException("Matrix dimensions mismatch in static method: Matrix sub(Matrix mat1, Matrix mat2): ("
					+ mat1.getRows() + "," + mat1.getCols() + ") - (" + mat2.getRows() + "," + mat2.getCols() + ") dimensions must be the same");
			
		double[][] resultData = new double[mat1.getRows()][mat1.getCols()];
		for (int i = 0; i < mat1.getRows(); i++)
			for (int j = 0; j < mat1.getCols(); j++)
				resultData[i][j] = mat1.get(i, j) - mat2.get(i, j);
		
		return new Matrix(resultData);
	}
	
	public Matrix sub(double scalar) {
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				data[i][j] -= scalar;
		return this;
	}
	
	public static Matrix sub(Matrix mat, double scalar) {
		
		double[][] resultData = new double[mat.getRows()][mat.getCols()];
		for (int i = 0; i < mat.getRows(); i++)
			for (int j = 0; j < mat.getCols(); j++)
				resultData[i][j] = mat.get(i, j) - scalar;
		
		return new Matrix(resultData);
	}
	
	public double sum() {
		
		return this.flatten().sum();
	}
	
	public double average() {
		
		return this.flatten().average();
	}
	
	public Matrix abs() {
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				set(i, j, Math.abs(get(i, j)));
		return this;
	}
	
	public static Matrix abs(Matrix mat) {
		
		double[][] resultData = new double[mat.getRows()][mat.getCols()];
		for (int i = 0; i < mat.getRows(); i++)
			for (int j = 0; j < mat.getCols(); j++)
				resultData[i][j] = Math.abs(mat.get(i, j));
		
		return new Matrix(resultData);
	}
	
	public double dot(Matrix mat) {
		
		if (getRows() != mat.getRows() || getCols() != mat.getCols())
			throw new RuntimeException("Matrix dimensions mismatch in dynamic method: double dot(Matrix mat): ("
					+ getRows() + "," + getCols() + ") x (" + mat.getRows() + "," + mat.getCols() + ") dimensions must be the same");
			
		Vector vec1 = this.flatten();
		Vector vec2 = mat.flatten();
		
		return vec1.dot(vec2);
	}
	
	public Matrix mult(Matrix mat) {
		
		if (getCols() != mat.getRows())
			throw new RuntimeException("Matrix dimensions mismatch in dynamic method: Matrix mult(Matrix mat):"
					+ " (" + getRows() + "," + getCols() + ") * (" + mat.getRows() + "," + mat.getCols() + ") dimensions must agree");
			
		double[][] resultData = new double[getRows()][mat.getCols()];
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < mat.getCols(); j++) {
				resultData[i][j] = 0.0f;
				for (int k = 0; k < getCols(); k++)
					resultData[i][j] += get(i, k) * mat.get(k, j);
			}
		
		set(resultData);	
		return this;
	}
	
	public static Matrix mult(Matrix mat1, Matrix mat2) {
		
		if (mat1.getCols() != mat2.getRows())
			throw new RuntimeException("Matrix dimensions mismatch in static method: Matrix mult(Matrix mat1, Matrix mat2):"
					+ " (" + mat1.getRows() + "," + mat1.getCols() + ") * (" + mat2.getRows() + "," + mat2.getCols() + ") dimensions must agree");
			
		double[][] resultData = new double[mat1.getRows()][mat2.getCols()];
		for (int i = 0; i < mat1.getRows(); i++)
			for (int j = 0; j < mat2.getCols(); j++) {
				resultData[i][j] = 0.0f;
				for (int k = 0; k < mat1.getCols(); k++)
					resultData[i][j] += mat1.get(i, k) * mat2.get(k, j);
			}
		
		return new Matrix(resultData);
	}
	
	public Matrix mult(double scalar) {
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				data[i][j] *= scalar;
		return this;
	}
	
	public static Matrix mult(Matrix mat, double scalar) {
		
		double[][] resultData = new double[mat.getRows()][mat.getCols()];
		for (int i = 0; i < mat.getRows(); i++)
			for (int j = 0; j < mat.getCols(); j++)
				resultData[i][j] = mat.get(i, j) * scalar;
		
		return new Matrix(resultData);
	}
	
	public Matrix multElementWise(Matrix mat) {
		
		if (getRows() != mat.getRows() || getCols() != mat.getCols())
			throw new RuntimeException("Matrix dimensions mismatch in dynamic method: Matrix multElementWise(Matrix mat):"
					+ " (" + getRows() + "," + getCols() + ") * (" + mat.getRows() + "," + mat.getCols() + ") dimensions must be the same");
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				data[i][j] *= mat.get(i, j);
		return this;
	}
	
	public static Matrix multElementWise(Matrix mat1, Matrix mat2) {
		
		if (mat1.getRows() != mat2.getRows() || mat1.getCols() != mat2.getCols())
			throw new RuntimeException("Matrix dimensions mismatch in static method: Matrix multElementWise(Matrix mat1, Matrix mat2):"
					+ " (" + mat1.getRows() + "," + mat1.getCols() + ") * (" + mat2.getRows() + "," + mat2.getCols() + ") dimensions must be the same");
			
		double[][] resultData = new double[mat1.getRows()][mat1.getCols()];
		for (int i = 0; i < mat1.getRows(); i++)
			for (int j = 0; j < mat1.getCols(); j++)
				resultData[i][j] = mat1.get(i, j) * mat2.get(i, j);
		
		return new Matrix(resultData);					
	}
	
	public Matrix div(double scalar) {
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				data[i][j] /= scalar;
		return this;
	}
	
	public static Matrix div(Matrix mat, double scalar) {
		
		double[][] resultData = new double[mat.getRows()][mat.getCols()];
		for (int i = 0; i < mat.getRows(); i++)
			for (int j = 0; j < mat.getCols(); j++)
				resultData[i][j] = mat.get(i, j) / scalar;
		
		return new Matrix(resultData);
	}
	
	public Matrix divElementWise(Matrix mat) {
		
		if (getRows() != mat.getRows() || getCols() != mat.getCols())
			throw new RuntimeException("Matrix dimensions mismatch in dynamic method: Matrix divElementWise(Matrix mat):"
					+ " (" + getRows() + "," + getCols() + ") / (" + mat.getRows() + "," + mat.getCols() + ") dimensions must be the same");
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				data[i][j] /= mat.get(i, j);
		return this;
	}
	
	public static Matrix divElementWise(Matrix mat1, Matrix mat2) {
		
		if (mat1.getRows() != mat2.getRows() || mat1.getCols() != mat2.getCols())
			throw new RuntimeException("Matrix dimensions mismatch in static method: Matrix divElementWise(Matrix mat1, Matrix mat2):"
					+ " (" + mat1.getRows() + "," + mat1.getCols() + ") / (" + mat2.getRows() + "," + mat2.getCols() + ") dimensions must be the same");
		
		double[][] resultData = new double[mat1.getRows()][mat1.getCols()];
		for (int i = 0; i < mat1.getRows(); i++)
			for (int j = 0; j < mat1.getCols(); j++)
				resultData[i][j] = mat1.get(i, j) / mat2.get(i, j);
		
		return new Matrix(resultData);		
	}
	
	public Matrix kronecker(Matrix mat) {
		
		double[][] resultData = new double[getRows() * mat.getRows()][getCols() * mat.getCols()];
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				for (int k = 0; k < mat.getRows(); k++)
					for (int l = 0; l < mat.getCols(); l++)
						resultData[i * mat.getRows() + k][j * mat.getCols() + l] = get(i, j) * mat.get(k, l);
		
		set(resultData);
		return this;
	}
	
	public static Matrix kronecker(Matrix mat1, Matrix mat2) {
		
		double[][] resultData = new double[mat1.getRows() * mat2.getRows()][mat1.getCols() * mat2.getCols()];
		for (int i = 0; i < mat1.getRows(); i++)
			for (int j = 0; j < mat1.getCols(); j++)
				for (int k = 0; k < mat2.getRows(); k++)
					for (int l = 0; l < mat2.getCols(); l++)
						resultData[i * mat2.getRows() + k][j * mat2.getCols() + l] = mat1.get(i, j) * mat2.get(k, l);
		
		return new Matrix(resultData);
	}
	
	public double determinant() {
		
		if (getRows() != getCols())
			throw new RuntimeException("The determinant() function is undefined for non square matrices");
		
		int n = getRows();
		if (n == 3)
			return get(0, 0) * get(1, 1) * get(2, 2) + get(0, 1) * get(1, 2) * get(2, 0) + get(1, 0) * get(2, 1) * get(0, 2)
					- get(0, 2) * get(1, 1) * get(2, 0) - get(1, 2) * get(2, 1) * get(0, 0) - get(0, 1) * get(1, 0) * get(2, 2);
		else if (n == 2)
			return get(0, 0) * get(1, 1) - get(1, 0) * get(0, 1);
		else if (n == 1)
			return get(0, 0);
		
		double result = 0;
		for (int i = 0; i < n; i++) {
			double [][] auxData = new double[n - 1][n - 1];
			
			int x = 0;
			for (int j = 1; j < n; j++) {
				
				int y = 0;
				for (int k = 0; k < n; k++)
					if (k != i)
						auxData[x][y++] = get(j, k);	
				x++;
			}
			
			Matrix aux = new Matrix(auxData);
			result += (i % 2 == 0 ? get(0, i) : - get(0, i)) * aux.determinant();
		}
		
		return result;
	}
	
	public Matrix adjoint() {
		
		if (getRows() != getCols())
			throw new RuntimeException("The adjoint() function is undefined for non square matrices");
		
		int n = getRows();
		double[][] resultData = new double[n][n];
		
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++) {
				double[][] auxData = new double[n - 1][n - 1];
				
				int x = 0;
				for (int k = 0; k < n; k++)
					if (k != i) {
						
						int y = 0;
						for (int l = 0; l < n; l++)
							if (l != j)
								auxData[x][y++] = get(k, l);
						x++;
					}
				
				Matrix aux = new Matrix(auxData);
				resultData[i][j] = (i + j) % 2 == 0 ? aux.determinant() : - aux.determinant();
			}
		
		return new Matrix(resultData);
	}
	
	public Matrix inverse() {
		
		if (getRows() != getCols())
			throw new RuntimeException("The inverse() function is undefined for non square matrices");
		
		double det = determinant();
		if (det == 0.0f)
			throw new RuntimeException("This matrix has no inverse (the determinant is zero)");
		
		Matrix result = Matrix.transpose(this.adjoint());
		result.div(det);
		
		return result;
	}
	
	public Matrix[] decomposeLU() {
		
		if (getRows() != getCols())
			throw new RuntimeException("The function decomposeLU() is undefined for non square matrices");
		
		int n = getRows();
		double[][] resultL = new double[n][n];
		double[][] resultU = new double[n][n];
		
		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++) {
				
				double sum = 0.0f;
				for (int k = 0; k < i; k++)
					sum += resultL[i][k] * resultU[k][j];
				
				resultU[i][j] = get(i, j) - sum;
			}
			
			for (int j = i; j < n; j++) {
				
				double sum = 0.0f;
				if (i == j)
					resultL[i][j] = 1.0f;
				else {
					for (int k = 0; k < i; k++)
						sum += resultL[j][k] * resultU[k][i];
					
					resultL[j][i] = (get(j, i) - sum) / resultU[i][i];
				}
			}
		}
		
		return new Matrix[] {new Matrix(resultL), new Matrix(resultU)};
	}
	
	public static Vector solveLU(Matrix mat, Vector b) {
		
		if (mat.getRows() != mat.getCols())
			throw new RuntimeException("The function solveLU() is undefined for non square matrices");
		
		int n = mat.getRows();
		if (b.getLength() != n)
			throw new RuntimeException("Vector length must be equal to matrix dimensions in solveLU()");
		
		Matrix[] factorized = mat.decomposeLU();
		
		Matrix L = factorized[0];
		Matrix U = factorized[1];
		
		double[] y = Vector.dataCopy(b.get());
		for (int i = 0; i < n; i++)
			for (int j = 0; j < i; j++)
				y[i] -= L.get(i, j) * y[j];
		
		double[] x = Vector.dataCopy(y);
		for (int i = n - 1; i >= 0; i--) {
			for (int j = n - 1; j > i; j--)
				x[i] -= U.get(i, j) * x[j];
			
			x[i] /= U.get(i, i);
		}
		
		return new Vector(x);
	}
	
	public static Vector solve(Matrix mat, Vector b) {
		
		if (mat.getRows() != mat.getCols())
			throw new RuntimeException("The function solve() is undefined for non square matrices");
		
		if (b.getLength() != mat.getRows())
			throw new RuntimeException("Vector length must be equal to matrix dimensions in solve()");
		
		return Vector.mult(mat.inverse(), b);
	}
	
	public static Vector jacobi(Matrix mat, Vector b, double eps) {
		
		if (mat.getRows() != mat.getCols())
			throw new RuntimeException("Must input a square matrix");
		
		int n = mat.getRows();
		if (n != b.getLength())
			throw new RuntimeException("Vector length must be equal to the matrix dimension");
		
		Vector current = Vector.divElementWise(b, mat.getDiagonal());
		Vector prev = new Vector(b.getLength());
		
		do {
			
			prev = current.copy();
			for (int i = 0; i < n; i++) {
				
				double val = b.get(i);
				for (int j = 0; j < n; j++)
					if (i != j)
						val -= mat.get(i, j) * prev.get(j);
				
				val /= mat.get(i, i);
				current.set(i, val);
			}
		} while (Vector.sub(current, prev).norm() > eps);
		
		return current;
	}
	
	public static Vector gaussSeidel(Matrix mat, Vector b, double eps) {
		
		if (mat.getRows() != mat.getCols())
			throw new RuntimeException("Must input a square matrix");
		
		int n = mat.getRows();
		if (n != b.getLength())
			throw new RuntimeException("Vector length must be equal to the matrix dimension");
		
		Vector current = Vector.divElementWise(b, mat.getDiagonal());
		Vector prev;
		
		do {
			
			prev = current.copy();
			for (int i = 0; i < n; i++) {
				
				double val = b.get(i);
				for (int j = 0; j < i; j++)
					val -= mat.get(i, j) * current.get(j);
				
				for (int j = i + 1; j < n; j++)
					val -= mat.get(i, j) * prev.get(j);
				
				val /= mat.get(i, i);
				current.set(i, val);
			}
			
		} while (Vector.sub(current, prev).norm() > eps);
		
		return current;
	}
	
	public double maxRowNormSq() {
		
		double[] norms = new double[getRows()];
		for (int i = 0; i < getRows(); i++)
			norms[i] = this.fromRow(i).normSq();
		
		return Vector.fromArray(norms).max();
	}
	
	public double maxColNormSq() {
		
		double[] norms = new double[getCols()];
		for (int i = 0; i < getCols(); i++)
			norms[i] = this.fromCol(i).normSq();
		
		return Vector.fromArray(norms).max();
	}
	
	public double maxRowNorm() {
		
		return (double) Math.sqrt(maxRowNormSq());
	}
	
	public double maxColNorm() {
		
		return (double) Math.sqrt(maxColNormSq());
	}
	
	public double superNormSq() {
		
		return this.flatten().normSq();
	}
	
	public double superNorm() {
		
		return this.flatten().norm();
	}
	
	public double max() {
		
		return this.flatten().max();
	}
	
	public double min() {
		
		return this.flatten().min();
	}
	
	public int[] indexMax() {
		int[] cols = new int[getRows()];
		Vector maxes = new Vector(getRows());
		
		for (int i = 0; i < getRows(); i++) {
			Vector row = getRow(i);
			maxes.set(i, row.max());
			cols[i] = row.indexMax();
		}
		
		int maxIndex = maxes.indexMax();
		
		int maxI = maxIndex;
		int maxJ = cols[maxIndex];
		
		return new int[] {maxI, maxJ};
	}
	
	public int[] indexMin() {
		int[] cols = new int[getRows()];
		Vector mins = new Vector(getRows());
		
		for (int i = 0; i < getRows(); i++) {
			Vector row = getRow(i);
			mins.set(i, row.min());
			cols[i] = row.indexMin();
		}
		
		int minIndex = mins.indexMin();
		
		int minI = minIndex;
		int minJ = cols[minIndex];
		
		return new int[] {minI, minJ};
	}
	
	public Matrix transpose() {
		
		double[][] resultData = new double[getCols()][getRows()];
		for (int i = 0; i < getCols(); i++)
			for (int j = 0; j < getRows(); j++)
				resultData[i][j] = get(j, i);
		
		set(resultData);
		return this;
	}
	
	public static Matrix transpose(Matrix mat) {
		
		double[][] resultData = new double[mat.getCols()][mat.getRows()];
		for (int i = 0; i < mat.getCols(); i++)
			for (int j = 0; j < mat.getRows(); j++)
				resultData[i][j] = mat.get(j, i);
		
		return new Matrix(resultData);
	}
	
	public Vector flatten() {
		
		double[] resultData = new double[getRows() * getCols()];
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				resultData[i * getCols() + j] = get(i, j);
		
		return new Vector(resultData);
	}
	
	public Matrix reshape(int x, int y) {
		
		if (x * y != getRows() * getCols())
			throw new RuntimeException("Dimension factor mismatch in dynamic method Matrix resize(int x, int y). Make sure x * y == getRows() * getCols()");
			
		double[][] resultData = new double[x][y];
		Vector vec = this.flatten();
		
		for (int i = 0; i < x; i++)
			for (int j = 0; j < y; j++)
				resultData[i][j] = vec.get(i * y + j);
		
		set(resultData);
		return this;
	}
	
	public static Matrix reshape(Matrix mat, int x, int y) {
		
		if (x * y != mat.getRows() * mat.getCols())
			throw new RuntimeException("Dimension factor mismatch in static method Matrix resize(int x, int y). Make sure x * y == getRows() * getCols()");
			
		double[][] resultData = new double[x][y];
		Vector vec = mat.flatten();
		
		for (int i = 0; i < x; i++)
			for (int j = 0; j < y; j++)
				resultData[i][j] = vec.get(i * y + j);
		
		return new Matrix(resultData);
	}
	
	public static Matrix convolve(Matrix mat1, Matrix mat2, int stride) {
		
		if (mat1.getRows() < mat2.getRows() || mat1.getCols() < mat2.getCols())
			throw new RuntimeException("Filter dimensions exceed convolved matrix");
			
		int X = mat1.getRows() - mat2.getRows();
		int Y = mat1.getCols() - mat2.getCols();
		
		X /= stride;
		Y /= stride;
		
		X++;
		Y++;
		
		double[][] resultData = new double[X][Y];
		for (int i = 0; i < X; i++) {
			int x = i * stride;
			
			for (int j = 0; j < Y; j++) {
				int y = j * stride;
				
				Matrix portion = mat1.getPortion(x, y, mat2.getRows(), mat2.getCols());
				resultData[i][j] = mat2.dot(portion);
			}
		}
				
		return new Matrix(resultData);
	}
	
	public static int[] afterConv(int[] inShape, int[] filtShape, int stride) {
		
		if (inShape.length < 2 || filtShape.length < 2)
			throw new RuntimeException("Only 2D convolutions allowed in afterConv()");
		
		int[] result = new int[inShape.length];
		
		result[0] = inShape[0] - filtShape[0];
		result[1] = inShape[1] - filtShape[1];
		
		result[0] /= stride;
		result[1] /= stride;
		
		result[0]++;
		result[1]++;
		
		return result;
	}
	
	public static int[] toMaintainConvDim(int[] inShape, int[] filtShape, int stride) {
		
		if (inShape.length < 2 || filtShape.length < 2)
			throw new RuntimeException("Only 2D convolutions allowed in toMaintainConvDim()");
		
		int[] result = new int[inShape.length];
		
		result[0] = filtShape[0] + (inShape[0] - 1) * (stride - 1);
		result[1] = filtShape[1] + (inShape[1] - 1) * (stride - 1);
		
		result[0] /= 2;
		result[1] /= 2;
				
		return result;
	}
	
	public Matrix zeroPad(int verLayers, int horLayers) {
		
		set(Matrix.zeroPad(this, verLayers, horLayers).get());
		return this;
	}
	
	public static Matrix zeroPad(Matrix mat, int verLayers, int horLayers) {
		
		if (horLayers == 0 && verLayers == 0)
			return mat;
		
		Matrix appended = mat;
				
		if (verLayers > 0) {
			int medium = verLayers;
			Matrix[] vertical = new Matrix[2 * verLayers + 1];
			
			for (int i = 0; i < vertical.length; i++)
				if (i == medium)
					vertical[i] = appended;
				else
					vertical[i] = new Matrix(1, appended.getCols());
			
			appended = Matrix.appendVer(vertical);
		}
				
		if (horLayers > 0) {
			int medium = horLayers;
			Matrix[] horizontal = new Matrix[2 * horLayers + 1];
			
			for (int i = 0; i < horizontal.length; i++)
				if (i == medium)
					horizontal[i] = appended;
				else
					horizontal[i] = new Matrix(appended.getRows(), 1);
			
			appended = Matrix.appendHor(horizontal);
		}
		
		return appended;
	}
	
	public Matrix unPad(int verLayers, int horLayers) {
		
		set(unPad(this, verLayers, horLayers).get());
		return this;
	}
	
	public static Matrix unPad(Matrix mat, int verLayers, int horLayers) {
		if (mat.getRows() < 2 * verLayers || mat.getCols() < 2 * horLayers)
			throw new RuntimeException("Matrix dimensions must be greater than removed layers in unPad()");
		
		return mat.getPortion(verLayers, horLayers, mat.getRows() - 2 * verLayers, mat.getCols() - 2 * verLayers);
	}
	
	public static Matrix maxPooling(Matrix mat, int stride) {
		
		if (stride <= mat.getRows() && stride <= mat.getCols()) {
			
			int X = mat.getRows() / stride;
			int Y = mat.getCols() / stride;
			
			double[][] resultData = new double[X][Y];
			
			for (int i = 0; i < X; i++) {
				int x = i * stride;
				
				for (int j = 0; j < Y; j++) {
					int y = j * stride;
					
					resultData[i][j] = mat.getPortion(x, y, stride, stride).max();
				}
			}
			
			return new Matrix(resultData);
		} else throw new RuntimeException("Result matrix will have null dimension in maxPooling");
	}
	
	public static int[] afterPool(int[] inShape, int stride) {
		
		if (inShape.length < 2)
			throw new RuntimeException("Only 2D pooling allowed in afterPool()");
		
		int[] result = new int[inShape.length];
		
		result[0] = inShape[0] / stride;
		result[1] = inShape[1] / stride;
		
		return result;
	}
	
	public static Matrix averagePooling(Matrix mat, int stride) {
		
		if (stride <= mat.getRows() && stride <= mat.getCols()) {

			int X = mat.getRows() / stride;
			int Y = mat.getCols() / stride;
			
			double[][] resultData = new double[X][Y];
			
			for (int i = 0; i < X; i++) {
				int x = i * stride;
				
				for (int j = 0; j < Y; j++) {
					int y = j * stride;
					
					resultData[i][j] = mat.getPortion(x, y, stride, stride).average();
				}
			}
			
			return new Matrix(resultData);
		} else throw new RuntimeException("Result matrix will have null dimension in averagePooling");
	}
	
	public static Matrix appendHor(Matrix[] list) {
		
		if (list.length == 1)
			return list[0];
		else if (list.length == 0)
			throw new RuntimeException("Empty list in Matrix.append()");
		else {
		
			int rows = list[0].getRows();
			int cols = 0;
			
			for (Matrix mat : list)
				if (mat.getRows() != rows)
					throw new RuntimeException("To append, all matrices must have the same number of rows or cols, depending on the direction");
				else
					cols += mat.getCols();
				
			double[][] resultData = new double[rows][cols];
			for (int i = 0; i < rows; i++) {
				
				Vector[] vecList = new Vector[list.length];
				for (int j = 0; j < list.length; j++)
					vecList[j] = list[j].fromRow(i);
				
				resultData[i] = Vector.append(vecList).get();
			}
			
			return new Matrix(resultData);
		}
	}
	
	public static Matrix appendVer(Matrix[] list) {
		
		if (list.length == 1)
			return list[0];
		
		Matrix[] transList = new Matrix[list.length];
		for (int i = 0; i < list.length; i++)
			transList[i] = Matrix.transpose(list[i]);
		
		return Matrix.transpose(appendHor(transList));
	}
	
	public Matrix[] divideHor(int total) {
		
		if (total == 1)
			return new Matrix[] {this};
		
		Matrix[] result = new Matrix[total];
		Vector[][] portions = new Vector[getRows()][];
		for (int i = 0; i < getRows(); i++)
			portions[i] = this.fromRow(i).divide(total);
		
		for (int i = 0; i < total; i++) {
			Matrix[] toAppend = new Matrix[getRows()];
			for (int j = 0; j < getRows(); j++)
				toAppend[j] = portions[j][i].reshape(1, portions[0][i].getLength());
			
			result[i] = Matrix.appendVer(toAppend);
		}
		
		return result;
	}
	
	public Matrix[] divideVer(int total) {
		
		if (total == 1)
			return new Matrix[] {this};
		
		Matrix trans = transpose(this);
		Matrix[] result = trans.divideHor(total);
		
		for (Matrix mat : result)
			mat.transpose();
		
		return result;
	}
	
	public Matrix rotate(int times) {
		
		Matrix rotated = this;
		if (times > 0)
			for (int i = 0; i < times; i++)
				rotated = Matrix.rotateOnce(rotated, true);
		else if (times < 0)
			for (int i = 0; i < - times; i++)
				rotated = Matrix.rotateOnce(rotated, false);
		
		set(rotated.get());
		return this;
	}
	
	public static Matrix rotate(Matrix mat, int times) {
		
		Matrix rotated = mat;
		while (times > 3)
			times -= 4;
		while (times < - 3)
			times += 4;
		
		if (times > 0)
			for (int i = 0; i < times; i++)
				rotated = Matrix.rotateOnce(rotated, true);
		else if (times < 0)
			for (int i = 0; i < - times; i++)
				rotated = Matrix.rotateOnce(rotated, false);
		else
			return mat.copy();
		
		return rotated;
	}
	
	private static Matrix rotateOnce(Matrix mat, boolean counterClock) {
		
		Matrix[] matrixList = new Matrix[mat.getRows()];
		for (int i = 0; i < mat.getRows(); i++) {
			Vector extract = mat.fromRow(i);
			if (counterClock) {
				extract.reverse();
				matrixList[i] = extract.reshape(mat.getCols(), 1);
			} else
				matrixList[mat.getRows() - 1 - i] =extract.reshape(mat.getCols(), 1);
		}
		
		return Matrix.appendHor(matrixList);
	}
	
	public static Matrix rotate2D(double angle) {
		
		double cos = (double) Math.cos(angle);
		double sin = (double) Math.sin(angle);
		return new Matrix(new double[][] {
				{cos, - sin},
				{sin, cos}
		});
	}
	
	public static Matrix rotateX(double angle) {
		
		double cos = (double) Math.cos(angle);
		double sin = (double) Math.sin(angle);
		return new Matrix(new double[][] {
			{1, 0, 0},
			{0, cos, - sin},
			{0, sin, cos}
	});
	}
	
	public static Matrix rotateY(double angle) {
		
		double cos = (double) Math.cos(angle);
		double sin = (double) Math.sin(angle);
		return new Matrix(new double[][] {
			{cos, 0, sin},
			{0, 1, 0},
			{ - sin, 0, cos}
	});
	}

	public static Matrix rotateZ(double angle) {
	
	double cos = (double) Math.cos(angle);
	double sin = (double) Math.sin(angle);
	return new Matrix(new double[][] {
		{cos, - sin, 0},
		{sin, cos, 0},
		{0, 0, 1}
	});
	}
		
	public Matrix copy() {
		
		return new Matrix(this.toArray());
	}
	
	public boolean isNaN() {
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				if (Double.isNaN(get(i, j)))
					return true;
		
		return false;
	}
	
	
	public Matrix(double[][] data) {set(data);}
	public int getRows() {return rows;}
	public int getCols() {return cols;}
	public int getSize() {return cols * rows;}
	public double[][] get(){return data;}
	public double get(int i, int j) {
		if (i >= getRows() || j >= getCols()) 
			throw new RuntimeException("Index exceeds matrix dimensions in Matrix get(int i, int j): (" 
					+ i + "," + j + ") -> (" + getRows() + "," + getCols() + ")");
		return data[i][j];		
	}
	
	public Vector getRow(int i) {
		if (i >= getRows())
			throw new RuntimeException("Index exceeds matrix dimensions in Vector getRow(int i): (rows: "
					+ getRows() + " index: " + i);
		return new Vector(data[i]);		
	}
	
	public Vector getDiagonal() {
		
		if (getRows() != getCols())
			throw new RuntimeException("Must be a square matrix");
		
		int n = getRows();
		double[] resultData = new double[n];
		for (int i = 0; i < n; i++)
			resultData[i] = get(i, i);
		
		return new Vector(resultData);
	}
	
	public Matrix getPortion(int startX, int startY, int deltaX, int deltaY) {
		
		if (deltaX <= 0 || deltaY <= 0)
			throw new RuntimeException("Selected portion dimensions are less or equal than zero in getPortion()");
			
		double[][] resultData = new double[deltaX][deltaY];
		
		int endX = startX + deltaX;
		int endY = startY + deltaY;
		
		for (int i = startX; i < endX; i++)
			for (int j = startY; j < endY; j++)
				resultData[i - startX][j - startY] = get(i, j);
		
		return new Matrix(resultData);
	}
	
	public Matrix setPortion(Matrix mat, int startX, int startY) {
		
		int endX = startX + mat.getRows();
		int endY = startY + mat.getCols();
		
		if (endX > getRows() && endY > getCols())
			throw new RuntimeException("Specified location exceeds matrix dimensions in setPortion()");
		
		for (int i = startX; i < endX; i++)
			for (int j = startY; j < endY; j++)
				set(i, j, mat.get(i - startX, j - startY));
		return this;
	}
	
	public Matrix set(int i, int j, double scalar) {
		
		if (i >= getRows() && j >= getCols())
			throw new RuntimeException("Index exceeds matrix dimensions in Matrix set(int i, int j, double scalar): (" 
					+ i + "," + j + ") -> (" + getRows() + "," + getCols() + ")");
		data[i][j] = scalar;
		return this;
	}
	
	public Matrix set(double[][] input) {
		
		data = input; rows = input.length; cols = input[0].length;
		return this;
	}
	
	public Matrix setRow(double[] input, int i) {
		
		if (input.length != getCols() || i >= getRows())
			throw new RuntimeException("Dimensions mismatch in Matrix setRow(double[] input, int i): " 
					+ input.length + "!=" + getCols() + " or" + i + ">=" + getRows());
			
		data[i] = Vector.dataCopy(input);	
		return this;
	}
	
	public Matrix setCol(double[] input, int j) {
		
		if (input.length == getRows() && j < getCols())
			throw new RuntimeException("Dimensions mismatch in Matrix setCol(double[] input, int j): " 
					+ input.length + "!=" + getRows() + " or" + j + ">=" + getCols());
			
		for (int i = 0; i < getRows(); i++)
			data[i][j] = input[i];	
		return this;
	}
	
	public Matrix setAll(double scalar) {
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				data[i][j] = scalar;
		return this;
	}
	
	public Matrix setAllBut(int x, int y, double scalar) {
		
		if (x >= getRows() || y >= getCols())
			throw new RuntimeException("Index exceeds matrix dimensions in Matrix setAllBut()");
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				if (i != x || j != y)
					data[i][j] = scalar;
		return this;
	}
	
	public Matrix setAsProbable(double scalar, double rate) {
		
		for (int i = 0; i < getRows(); i++)
			for (int j = 0; j < getCols(); j++)
				if (rand.nextDouble() < rate)
					data[i][j] = scalar;
		return this;
	}
	
	public static Matrix setAsProbable(Matrix mat, double scalar, double rate) {
		
		double[][] resultData = new double[mat.getRows()][mat.getCols()];
		for (int i = 0; i < mat.getRows(); i++)
			for (int j = 0; j < mat.getCols(); j++)
				if (rand.nextDouble() < rate)
					resultData[i][j] = scalar;
		
		return new Matrix(resultData);
	}
	
	public double[][] toArray() {
		
		return Matrix.dataCopy(data);
	}
	
	public static Matrix fromArray(double[][] input) {
		
		return new Matrix(Matrix.dataCopy(input));
	}
	
	public Vector fromRow(int i) {
		
		if (i >= getRows())
			throw new RuntimeException("Index exceeds matrix dimensions in Vector fromRow(int i): (rows: "
					+ getRows() + " index: " + i);
		
		return new Vector(Vector.dataCopy(data[i]));
	}
	
	public Vector fromCol(int j) {
		
		if (j >= getCols())
			throw new RuntimeException("Index exceeds matrix dimensions in Vector fromCol(int j): (cols: "
					+ getCols() + " index: " + j);
		
		double[] result = new double[getRows()];
		for (int i = 0; i < getRows(); i++)
			result[i] = get(i, j);
		
		return new Vector(result);			
	}
	
	public Matrix print() {
		
		for (int i = 0; i < getRows(); i++)
			System.out.println(Arrays.toString(data[i]));
		System.out.println("----");
		return this;
	}
}

