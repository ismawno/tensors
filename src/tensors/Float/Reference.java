package tensors.Float;

public class Reference {

	
	public Reference() {
		
		
	}
	
	public void printVector() {
		
		System.out.println("REFERENCE of the class Vector:");
		System.out.println("Vector(int length): Creates a vector specifying its length and initialized to 0");
		System.out.println("Vector(float[] data): Creates a vector from an array");
		
		System.out.println("------");
		
		System.out.println("-Dyn int getLength(): Returns the vector length");
		System.out.println("-Dyn float get(int i): Returns the vector element i");
		System.out.println("-Dyn float[] get(): Returns by reference the data of the vector");
		System.out.println("-Dyn void set(float[] input): Sets the data of the array to input");
		System.out.println("-Dyn void setAll(float scalar): Sets all elements to the value scalar");
		System.out.println("-Dyn float[] toArray(): Returns a copy of the vector data");
		System.out.println("-Stat Vector fromArray(float[] input): Returns a vector with a copy of input as data");
		System.out.println("-Dyn float[] fromVector(): Returns a copy of the vector data");
		System.out.println("-Dyn void randomize(float deviation): Randomize with the specified deviation");
		System.out.println("You can pass two bounds as inputs to get linearly distributed values between them");
		System.out.println("-Dyn float max()/min(): Returns max/min element");
		System.out.println("-Dyn/Stat add()/sub(): Can be adressed in many ways: ");
		System.out.println("----Dynamic----");
		System.out.println("-Dyn void add()/sub(): If the input is a scalar, its value will be added or substracted to all elements");
		System.out.println("Otherwise, if it is a vector, it will add such vector. They must be the same length");
		System.out.println("----Static----");
		System.out.println("-Stat Vector add()/sub(): First input is always a Vector such that the function returns a new Vector that is the result of the operation.");
		System.out.println("If the second input is a scalar, it will add or substract the value to all elements");
		System.out.println("Otherwise, if it is a Vector, the result will be the addition or substraction between both");
		System.out.println("-Dyn float dot(Vector vec): Returns the dot product");
	}
}
