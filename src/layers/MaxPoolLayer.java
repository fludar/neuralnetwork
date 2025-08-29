package layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer {

    private int stepSize;
    private int windowSize;

    private int inputLength;
    private int inputRows;
    private int inputCols;

    List<int[][]> lastMaxRow;
    List<int[][]> lastMaxCol;

    public MaxPoolLayer(int stepSize, int windowSize, int inputLength, int inputRows, int inputCols) {
        this.stepSize = stepSize;
        this.windowSize = windowSize;
        this.inputLength = inputLength;
        this.inputRows = inputRows;
        this.inputCols = inputCols;
    }

    public List<double[][]> maxPoolForwardPass(List<double[][]> input) {
        List<double[][]> out = new ArrayList<>();

        lastMaxRow = new ArrayList<>();
        lastMaxCol = new ArrayList<>();

        for(int l = 0; l < input.size(); l++){
            out.add(pool(input.get(l)));
        }

        return out;
    }

    public double[][] pool(double[][] input){

        double[][] out = new double[getOutputRows()][getOutputCols()];

        int[][] maxRows = new int[getOutputRows()][getOutputCols()];
        int[][] maxCols = new int[getOutputRows()][getOutputCols()];

        for(int r = 0; r < getOutputRows(); r += stepSize){
            for(int c = 0; c < getOutputCols(); c += stepSize){

                double max = 0.0;
                maxRows[r][c] = -1;
                maxCols[r][c] = -1;
                
                for(int x = 0; x < windowSize; x++){
                    for(int y = 0; y < windowSize; y++){
                        if(max < input[r + x][c + y]){
                            max = input[r + x][c + y];
                            maxRows[r][c] = r + x;
                            maxCols[r][c] = c + y;
                        }
                    }   
                }

                out[r][c] = max;

            }
        }

        lastMaxRow.add(maxRows);
        lastMaxCol.add(maxCols);

        return out;
    }
    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputPool = maxPoolForwardPass(input);
        return nextLayer.getOutput(outputPool);
    }

    @Override
    public double[] getOutput(double[] input) {
       List<double[][]> matrixList = vectorToMatrix(input, inputLength,inputRows, inputCols);
       return getOutput(matrixList);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        List<double[][]> matrixList = vectorToMatrix(dLdO, getOutputLength(), getOutputRows(), getOutputCols());
        backPropagation(matrixList);
    }
    
    @Override
    public void backPropagation(List<double[][]> dLdO) {
        List<double[][]> dXdL = new ArrayList<>();
        int l = 0;

        for(double[][] array : dLdO) {
            double[][] error = new double[inputRows][inputCols];
    
            for(int r = 0; r < getOutputRows(); r++) {
                for(int c = 0; c < getOutputCols(); c++) {
                    int max_i = lastMaxRow.get(l)[r][c];
                    int max_j = lastMaxCol.get(l)[r][c];

                    if(max_i != -1){
                        error[max_i][max_j] += array[r][c];
                    }
                }
            }
            dXdL.add(error);
            l++;
        }
        
        if(previousLayer != null) {
            previousLayer.backPropagation(dXdL);
        }

    }

    @Override
    public int getOutputLength() {
        return inputLength;
    }
    
    @Override
    public int getOutputRows() {
        return (inputRows - windowSize) / stepSize + 1;
    }
    
    @Override
    public int getOutputCols() {
        return (inputCols - windowSize) / stepSize + 1;
    }
    
    @Override
    public int getOutputElements() {
        return inputLength * getOutputRows() * getOutputCols();
    }
    
}
