package layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvolutionLayer extends Layer {

    private long SEED;

    private List<double[][]> filters;
    private int filterSize;
    private int stepSize;

    private int inputLength;
    private int inputRows;
    private int inputCols;

    private void generateRandomFilters(int numFilters) {
        List<double[][]> genFilters = new ArrayList<>();
        Random random = new Random(SEED);

        for (int n = 0; n < numFilters; n++) {
            double[][] newFilter = new double[filterSize][filterSize];
            
            for (int i = 0; i<filterSize; i++) {
                for (int j = 0; j<filterSize; j++) {
                    double value = random.nextGaussian();
                    newFilter[i][j] = value;
                }
            }

            filters.add(newFilter);
        
        }
        
        filters = genFilters;
    }

    public ConvolutionLayer(int filterSize, int stepSize, int inputLength, int inputRows, int inputCols, long SEED, int numFilters) {
        this.filterSize = filterSize;
        this.stepSize = stepSize;
        this.inputLength = inputLength;
        this.inputRows = inputRows;
        this.inputCols = inputCols;
        this.SEED = SEED;

        generateRandomFilters(numFilters);
    }

    public List<double[][]> convolutionForwardPass(List<double[][]> list) {
        List<double[][]> output = new ArrayList<>();

        for (int m = 0; m < list.size(); m++) {
            for (double[][] filter : filters) {
                output.add(convolve(list.get(m), filter, stepSize));
            }
        }

        return output;
    }

    private double[][] convolve(double[][] input, double[][] filter, int stepsize) {
        int outputRows = (input.length - filter.length) / stepsize + 1;
        int outputCols = (input[0].length - filter[0].length) / stepsize + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outputRows][outputCols]; 

        int outRow = 0;
        int outCol;

        for (int i = 0; i <= inRows - fRows; i += stepsize) {

            outCol = 0;

            for (int j = 0; j <= inCols - fCols; j += stepsize) {

                double sum = 0.0;

                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        sum += input[i + x][j + y] * filter[x][y];
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;

            }

            outRow++;
        
        }

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> output = convolutionForwardPass(input);

        return nextLayer.getOutput(output);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixInput = vectorToMatrix(input, inputLength, inputRows, inputCols);
        return getOutput(matrixInput);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'backPropagation'");
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'backPropagation'");
    }

    @Override
    public int getOutputLength() {
        return filters.size() * inputLength;
    }

    @Override
    public int getOutputRows() {
        return (inputRows - filterSize) / stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (inputCols - filterSize) / stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputLength() * getOutputRows() * getOutputCols();
    }

}
