package layers;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {
    protected Layer nextLayer;
    protected Layer previousLayer;

    public Layer getNextLayer() {
        return nextLayer;
    }

    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public Layer getPreviousLayer() {
        return previousLayer;
    }

    public void setPreviousLayer(Layer previousLayer) {
        this.previousLayer = previousLayer;
    }

    public abstract double[] getOutput(List<double[][]> input);
    public abstract double[] getOutput(double[][] input);
    
    public abstract void backPropagation(double[] dLdO);
    public abstract void backPropagation(List<double[]> dLdO);

    public abstract int getOutputLength();
    public abstract int getOutputRows();
    public abstract int getOutputCols();
    public abstract int getOutputElements();

    public double[] matrixToVector(List<double[][]> input){

        int length = input.size();
        int rows = input.get(0).length;
        int cols = input.get(0)[0].length;

        double[] vector = new double[length * rows * cols];

        int k = 0;
        for(int l = 0; l < length; l++){
            for(int r = 0; r < rows; r++){
                for(int c = 0; c < cols; c++){
                    vector[k] = input.get(l)[r][c];
                    k++;
                }
            }
        }

        return vector;
        
    }

    List<double[][]> vectorToMatrix(double[] input, int length, int rows, int cols) {
        List<double[][]> out = new ArrayList<>();

        int i = 0;
        for(int l = 0; l < length; l++){

            double[][] mat = new double[rows][cols];

            for(int r = 0; r < rows; r++){
                for(int c = 0; c < cols; c++){
                    mat[r][c] = input[i];
                    i++;
                }
            }

            out.add(mat);

        }
        
        return out;
    
    }

}
