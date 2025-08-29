package layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import static data.MatrixUtility.*;

public class ConvolutionLayer extends Layer {

    private long SEED;

    private List<double[][]> filters;
    private int filterSize;
    private int stepSize;

    private int inputLength;
    private int inputRows;
    private int inputCols;
    private double learningRate;

    private List<double[][]> lastInput;

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

            genFilters.add(newFilter);
        
        }
        
        filters = genFilters;
    }

    public ConvolutionLayer(int filterSize, int stepSize, int inputLength, int inputRows, int inputCols, long SEED,
    int numFilters, double learningRate) {
        this.filterSize = filterSize;
        this.stepSize = stepSize;
        this.inputLength = inputLength;
        this.inputRows = inputRows;
        this.inputCols = inputCols;
        this.SEED = SEED;
        this.learningRate = learningRate;

        generateRandomFilters(numFilters);
    }

    public List<double[][]> convolutionForwardPass(List<double[][]> list) {
        lastInput = list;

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

    private double[][] fullConvolve(double[][] input, double[][] filter) {
        int outputRows = (input.length + filter.length)  + 1;
        int outputCols = (input[0].length + filter[0].length)  + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outputRows][outputCols]; 

        int outRow = 0;
        int outCol;

        for (int i = -fRows + 1; i <= inRows; i++) {

            outCol = 0;

            for (int j = -fCols + 1; j <= inCols; j++) {

                double sum = 0.0;

                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        int inputRowIndex = i + x;
                        int inputColIndex = j + y;

                        if (inputRowIndex >= 0 && inputRowIndex < inRows && inputColIndex >= 0 && inputColIndex < inCols) {
                            sum += filter[x][y] * input[inputRowIndex][inputColIndex];
                        }
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;

            }

            outRow++;
        
        }

        return output;
    }

    public double[][] spaceArray(double[][] input) {
        if(stepSize == 1) {
            return input;
        }

        int outRows = (input.length - 1) * stepSize + 1;
        int outCols = (input[0].length - 1) * stepSize + 1;

        double[][] output = new double[outRows][outCols];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                output[i * stepSize][j * stepSize] = input[i][j];
            }
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
        List<double[][]> matrixInput = vectorToMatrix(dLdO, inputLength, inputRows, inputCols);
        backPropagation(matrixInput);
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        
        List<double[][]> filtersDelta = new ArrayList<>();
        List<double[][]>  dLdOPrevLayer = new ArrayList<>();

        for (int f = 0 ; f < filters.size(); f++) {
            filtersDelta.add(new double[filterSize][filterSize]);
        }

        for(int i = 0; i < lastInput.size(); i++) {

            double[][] errorForInput = new double[inputRows][inputCols];

            for(int f = 0; f < filters.size(); f++) {
                double[][] currFilter = filters.get(f);
                double[][] error = dLdO.get(i * filters.size() + f);
                
                double[][] spacedError = spaceArray(error);
                double[][] dLdF = convolve(lastInput.get(i), spacedError, 1);

                double[][] delta = multiply(dLdF, learningRate * -1);
                double[][] newTotalDelta = add(filtersDelta.get(f), delta);
                filtersDelta.set(f, newTotalDelta);
            
                double[][] flippedError = flipArrayHorizontally(flipArrayVertically(spacedError));
                errorForInput = add(errorForInput, fullConvolve(currFilter, flippedError));
            }

            dLdOPrevLayer.add(errorForInput);
        }

        for (int f = 0; f < filters.size(); f++) {
            double[][] modified = add(filtersDelta.get(f), filters.get(f));
            filters.set(f, modified);
        }

        if(previousLayer != null) {
            previousLayer.backPropagation(dLdOPrevLayer);
        }
    }

    public double[][] flipArrayHorizontally(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;

        double[][] flipped = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flipped[rows-i-1][j] = array[i][j];
            }
        }
        return flipped;
    }

    public double[][] flipArrayVertically(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;

        double[][] flipped = new double[rows][cols];
    
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flipped[i][cols-j-1] = array[i][j];
            }
        }
        return flipped;
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
