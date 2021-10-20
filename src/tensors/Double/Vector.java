package tensors.Double;

import java.io.Serializable;

import java.util.Arrays;
import java.util.Random;

public class Vector implements Serializable {
	
	private static final long serialVersionUID = 1L;
	
	private int length;
	private double[] data;
	
	private static final Random rand = new Random();
	
	public Vector(int length) {
		
		this.length = length;
		
		data = new double[getLength()];
	}
	
	public static double[] dataCopy(double[] prevData){
		
		double[] result = new double[prevData.length];
		for (int i = 0; i < prevData.length; i++)
			result[i] = prevData[i];
		
		return result;
	}
	
	public static Vector linspace(double a, double b, int N) {
		
		if (a == b) {
			Vector result = new Vector(N);
			result.setAll(a);
			return result;
		}
		
		double[] resultData = new double[N];
		
		for (int i = 0; i < N; i++) {
			
			double value = a + (b - a) * i / (N - 1);
			resultData[i] = value;
		}
		
		return new Vector(resultData);
	}
	
	public static int[] normLin(int a, int b) {
		
		if (b <= a)
			throw new RuntimeException("End point must be greater than start point in normLin()");
		
		int[] result = new int[b - a];
		int index = 0;
		for (int i = a; i < b; i++)
			result[index++] = i;
		
		return result;
	}
	
	public Vector randomize(double deviation) {
		
		for (int i = 0; i < getLength(); i++)
			set(i, (double) rand.nextGaussian() * deviation);
		return this;
	}
	
	public Vector randomize(double min, double max) {
		
		for (int i = 0; i < getLength(); i++)
			set(i, (double) rand.nextGaussian() * (max - min) + min);
		return this;
	}
	
	public Vector randomizeInt(int max) {
		
		for (int i = 0; i < getLength(); i++)
			set(i, rand.nextInt(max));
		return this;
	}
	
	public double max() {
		
		double comparator = get(0);
		for (int i = 1; i < getLength(); i++)
			if (comparator < get(i))
				comparator = get(i);
		
		return comparator;
	}
	
	public double min() {
		
		double comparator = get(0);
		for (int i = 1; i < getLength(); i++)
			if (comparator > get(i))
				comparator = get(i);
		
		return comparator;
	}
	
	public int indexMax() {
		
		int index = 0;
		double comparator = get(0);
		for (int i = 1; i < getLength(); i++)
			if (comparator < get(i)) {
				comparator = get(i);
				index = i;
			}
		
		return index;
	}
	
	public int indexMin() {
		
		int index = 0;
		double comparator = get(0);
		for (int i = 1; i < getLength(); i++)
			if (comparator > get(i)) {
				comparator = get(i);
				index = i;
			}
		
		return index;
	}
	
	public Vector abs() {
		
		for (int i = 0; i < getLength(); i++)
			set(i, Math.abs(get(i)));
		return this;
	}
	
	public static Vector abs(Vector vec) {
		
		double[] resultData = new double[vec.getLength()];
		for (int i = 0; i < vec.getLength(); i++)
			resultData[i] = Math.abs(vec.get(i));
		
		return new Vector(resultData);
	}
	
	public Vector add(Vector vec) {
		
		if (getLength() != vec.getLength())
			throw new RuntimeException("Vector dimensions mismatch in dynamic method: Vector add(Vector vec): ("	
					+ getLength() + ") + (" + vec.getLength() + ") dimensions must be the same");
		
		for (int i = 0; i < getLength(); i++)
			data[i] += vec.get(i);
		return this;
	}
	
	public static Vector add(Vector vec1, Vector vec2) {
		
		if (vec1.getLength() != vec2.getLength())
			throw new RuntimeException("Vector dimensions mismatch in static method: Vector add(Vector vec1, Vector vec2): ("	
					+ vec1.getLength() + ") + (" + vec2.getLength() + ") dimensions must be the same");
		
		double[] resultData = new double[vec1.getLength()];
		for (int i = 0; i < vec1.getLength(); i++)
			resultData[i] = vec1.get(i) + vec2.get(i);
		
		return new Vector(resultData);
	}
	
	public Vector add(double scalar) {
		
		for (int i = 0; i < getLength(); i++)
			data[i] += scalar;
		return this;
	}
	
	public static Vector add(Vector vec, double scalar) {
		
		double[] resultData = new double[vec.getLength()];
		for (int i = 0; i < vec.getLength(); i++)
			resultData[i] = vec.get(i) + scalar;
		
		return new Vector(resultData);
	}
	
	public Vector sub(Vector vec) {
		
		if (getLength() != vec.getLength())
			throw new RuntimeException("Vector dimensions mismatch in dynamic method: Vector sub(Vector vec): ("	
					+ getLength() + ") - (" + vec.getLength() + ") dimensions must be the same");
		
		for (int i = 0; i < getLength(); i++)
			data[i] -= vec.get(i);
		return this;
	}
	
	public static Vector sub(Vector vec1, Vector vec2) {
		
		if (vec1.getLength() != vec2.getLength())
			throw new RuntimeException("Vector dimensions mismatch in static method: Vector sub(Vector vec1, Vector vec2): ("	
					+ vec1.getLength() + ") - (" + vec2.getLength() + ") dimensions must be the same");
			
		double[] resultData = new double[vec1.getLength()];
		for (int i = 0; i < vec1.getLength(); i++)
			resultData[i] = vec1.get(i) - vec2.get(i);
		
		return new Vector(resultData);
	}
	
	public Vector sub(double scalar) {
		
		for (int i = 0; i < getLength(); i++)
			data[i] -= scalar;
		return this;
	}
	
	public static Vector sub(Vector vec, double scalar) {
		
		double[] resultData = new double[vec.getLength()];
		for (int i = 0; i < vec.getLength(); i++)
			resultData[i] = vec.get(i) - scalar;
		
		return new Vector(resultData);
	}
	
	public double sum() {
		
		double result = 0;
		for (int i = 0; i < getLength(); i++)
			result += get(i);
		
		return result;
	}
	
	public double average() {
		
		return this.sum() / getLength();
	}
	
	public double dot(Vector vec) {
		
		if (getLength() != vec.getLength())
			throw new RuntimeException("Vector dimensions mismatch in dynamic method: double dot(Vector vec): ("	
					+ getLength() + ") - (" + vec.getLength() + ") dimensions must be the same");
			
		double result = 0;
		for (int i = 0; i < getLength(); i++)
			result += get(i) * vec.get(i);
		
		return result;
	}
	
	public Vector mult(Matrix mat) {
		
		if (mat.getCols() != getLength())
			throw new RuntimeException("Tensor dimensions mismatch in dynamic method: Vector mult(Matrix mat):"
					+ " (" + mat.getRows() + "," + mat.getCols() + ") * (" + getLength() + "," + 1 + ") dimensions must agree");
			
		double[] resultData = new double[mat.getRows()];
		for (int i = 0; i < mat.getRows(); i++) {
			resultData[i] = 0.0f;
			for (int k = 0; k < mat.getCols(); k++)
				resultData[i] += mat.get(i, k) * get(k);
		}
		
		set(resultData);
		return this;
	}
	
	public static Vector mult(Matrix mat, Vector vec) {
		
		if (mat.getCols() != vec.getLength())
			throw new RuntimeException("Tensor dimensions mismatch in static method: Vector mult(Matrix mat, Vector vec):"
					+ " (" + mat.getRows() + "," + mat.getCols() + ") * (" + vec.getLength() + "," + 1 + ") dimensions must agree");
			
		double[] resultData = new double[mat.getRows()];
		for (int i = 0; i < mat.getRows(); i++) {
			resultData[i] = 0.0f;
			for (int k = 0; k < mat.getCols(); k++)
				resultData[i] += mat.get(i, k) * vec.get(k);
		}
		
		return new Vector(resultData);
	}
	
	public Matrix mult(Vector vec) {
		
		Matrix mat1 = this.reshape(getLength(), 1);
		Matrix mat2 = vec.reshape(1, vec.getLength());
		
		return Matrix.mult(mat1, mat2);
	}
	
	public Vector mult(double scalar) {
		
		for (int i = 0; i < getLength(); i++)
			data[i] *= scalar;
		return this;
	}
	
	public static Vector mult(Vector vec, double scalar) {
		
		double[] resultData = new double[vec.getLength()];
		for (int i = 0; i < vec.getLength(); i++)
			resultData[i] = vec.get(i) * scalar;
		
		return new Vector(resultData);
	}
	
	public Vector multElementWise(Vector vec) {
		
		if (getLength() != vec.getLength())
			throw new RuntimeException("Vector dimensions mismatch in dynamic method: Vector multElementWise(Vector vec):"
					+ " (" + getLength() + ") * (" + vec.getLength() + ") dimensions must be the same");
		
		for (int i = 0; i < getLength(); i++)
			data[i] *= vec.get(i);
		return this;
	}
	
	public static Vector multElementWise(Vector vec1, Vector vec2) {
		
		if (vec1.getLength() != vec2.getLength())
			throw new RuntimeException("Vector dimensions mismatch in static method: Vector multElementWise(Vector vec1, Vector vec2):"
					+ " (" + vec1.getLength() + ") * (" + vec2.getLength() + ") dimensions must be the same");
			
		double[] resultData = new double[vec1.getLength()];
		
		for (int i = 0; i < vec1.getLength(); i++)
			resultData[i] = vec1.get(i) * vec2.get(i);
		
		return new Vector(resultData);
	}
	
	public Vector div(double scalar) {
		
		for (int i = 0; i < getLength(); i++)
			data[i] /= scalar;
		return this;
	}
	
	public static Vector div(Vector vec, double scalar) {
		
		double[] resultData = new double[vec.getLength()];
		for (int i = 0; i < vec.getLength(); i++)
			resultData[i] = vec.get(i) / scalar;
		
		return new Vector(resultData);
	}
	
	public Vector divElementWise(Vector vec) {
		
		if (getLength() != vec.getLength())
			throw new RuntimeException("Vector dimensions mismatch in dynamic method: Vector divElementWise(Vector vec):"
					+ " (" + getLength() + ") * (" + vec.getLength() + ") dimensions must be the same");
		
		for (int i = 0; i < getLength(); i++)
			data[i] /= vec.get(i);
		return this;
	}
	
	public static Vector divElementWise(Vector vec1, Vector vec2) {
		
		if (vec1.getLength() != vec2.getLength())
			throw new RuntimeException("Vector dimensions mismatch in static method: Vector divElementWise(Vector vec1, Vector vec2):"
					+ " (" + vec1.getLength() + ") * (" + vec2.getLength() + ") dimensions must be the same");
			
		double[] resultData = new double[vec1.getLength()];
		
		for (int i = 0; i < vec1.getLength(); i++)
			resultData[i] = vec1.get(i) / vec2.get(i);
		
		return new Vector(resultData);
	}
	
	public double norm() {
		
		return (double) Math.sqrt(this.normSq());
	}
	
	public double normSq() {
		
		double result = 0;
		for (int i = 0; i < getLength(); i++)
			result += get(i) * get(i);
		
		return result;
	}
	
	public Matrix reshape(int x, int y) {
		
		double[][] resultData = new double[x][y];
		for (int i = 0; i < x; i++)
			for (int j = 0; j < y; j++)
				resultData[i][j] = get(i * y + j);
		
		return new Matrix(resultData);
	}
	
	public Vector[] divide(int total) {
		
		if (total == 1)
			return new Vector[] {this};
		
		Vector[] result;
		if (total > getLength()) {
			result = new Vector[getLength()];
			
			for (int i = 0; i < getLength(); i++)
				result[i] = new Vector(new double[] {get(i)});
			
			return result;
		}

		result = new Vector[total];
		
		int bigSpots = getLength() % total;
		int bigSpotsSize = getLength() / total + 1;
		
		int smallSpotsSize = getLength() / total;
		int smallSpots = total - bigSpots;
		
		int index1 = 0;
		int index2 = 0;
		
		for (int i = 0; i < bigSpots; i++) {
			
			Vector vec = new Vector(bigSpotsSize);
			for (int j = 0; j < bigSpotsSize; j++)
				vec.set(j, get(index1++));
			
			result[index2++] = vec;
		}
		
		for (int i = 0; i < smallSpots; i++) {
			
			Vector vec = new Vector(smallSpotsSize);
			for (int j = 0; j < smallSpotsSize; j++)
				vec.set(j, get(index1++));
			
			result[index2++] = vec;
		}
					
		return result;
	}
	
	public static Vector append(Vector[] list) {
		
		if (list.length == 1)
			return list[0];
		else if (list.length == 0)
			throw new RuntimeException("Empty list in Vector.append()");
		
		int totalLength = 0;
		int index = 0;
		
		Vector result;
		
		for (int i = 0; i < list.length; i++)
			totalLength += list[i].getLength();
		
		result = new Vector(totalLength);
		
		for (Vector vec : list)
			for (int j = 0; j < vec.getLength(); j++)
				result.set(index++, vec.get(j));
		
		return result;
	}
	
	public static Vector append(Matrix[] list) {
		
		Vector[] vecList = new Vector[list.length];
		for (int i = 0; i < list.length; i++)
			vecList[i] = list[i].flatten();
		
		return append(vecList);
	}
	
	public Vector reverse() {
		
		double[] resultData = new double[getLength()];
		for (int i = 0; i < getLength(); i++)
			resultData[i] = get(getLength() - 1 - i);
		
		set(resultData);
		return this;
	}
	
	public static Vector reverse(Vector vec) {
		
		double[] resultData = new double[vec.getLength()];
		for (int i = 0; i < vec.getLength(); i++)
			resultData[i] = vec.get(vec.getLength() - 1 - i);
		
		return new Vector(resultData);
	}
	
	public Vector copy() {
		
		return new Vector(this.toArray());
	}
	
	public boolean isNaN() {
		
		for (int i = 0; i < getLength(); i++)
			if (Double.isNaN(get(i)))
				return true;
		
		return false;
	}
	
	public Vector(double[] data) {set(data);}
	public int getLength() {return length;}
	public double get(int i) {
		if (i >= getLength())
			throw new RuntimeException("Index exceeds matrix length in double get(int i): " + i + ">=" + getLength());
		return data[i];
	}
	
	public double[] get() {return data;}
	public Vector set(int i, double scalar) {
		if (i >= getLength())
			throw new RuntimeException("Index exceeds matrix length in Vector set(int i): " + i + ">=" + getLength());
		
		data[i] = scalar;	
		return this;
	}
	
	public Vector set(double[] input) {
		
		data = input; length = input.length;
		return this;
	}
	
	public Vector setAll(double scalar) {
		
		for (int i = 0; i < getLength(); i++)
			data[i] = scalar;
		return this;
	}
	
	public Vector setAllBut(int x, double scalar) {
		
		if (x >= getLength())
			throw new RuntimeException("Index exceeds vector dimensions in setAllBut()");
		
		for (int i = 0; i < getLength(); i++)
			if (i != x)
				data[i] = scalar;
		return this;
	}
	
	public Vector setAsProbable(double scalar, double rate) {
		
		for (int i = 0; i < getLength(); i++)
			if (rand.nextDouble() < rate)
				data[i] = scalar;
		return this;
	}
	
	public static Vector setAsProbable(Vector vec, double scalar, double rate) {
		
		double[] resultData = new double[vec.getLength()];
		for (int i = 0; i < vec.getLength(); i++)
			if (rand.nextDouble() < rate)
				resultData[i] = scalar;
		
		return new Vector(resultData);
	}
	
	public double[] toArray() {
		
		return Vector.dataCopy(data);
	}
	
	public static Vector fromArray(double[] input) {
		
		return new Vector(Vector.dataCopy(input));
	}
	
	public Vector fromVector(int[] selection) {
		
		if (selection[selection.length - 1] >= getLength())
			throw new RuntimeException("Index array exceeds vector in fromVector()");
		
		double[] resultData = new double[selection.length];
		for (int i = 0; i < selection.length; i++)
			resultData[i] = get(selection[i]);
		
		return new Vector(resultData);
	}
	
	public Vector print() {
		
		System.out.println(Arrays.toString(get()));
		System.out.println("----");
		return this;
	}
}
