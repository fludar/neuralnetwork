package layers;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer {

    private long SEED;
    private final double leak = 0.01;

    private double[][] weights;
    private int inputLength;
    private int outputLength;
    private double learningRate;

    private double[] lastZ;
    private double[] lastX;

    public FullyConnectedLayer(int inputLength, int outputLength, long SEED, double learningRate) {
        this.inputLength = inputLength;
        this.outputLength = outputLength;
        this.SEED = SEED;
        this.learningRate = learningRate;
        
        weights = new double[inputLength][outputLength];
        setRandomWeights();
        
    }

    public double[] fullyConnectedForwardPass(double[] input) {
        double[] z = new double[outputLength];
        double[] out = new double[outputLength];

        for (int i = 0; i < inputLength; i++) {
            for (int j = 0; j < outputLength; j++) {
                z[j] += input[i] * weights[i][j];
        
            }
        }

        lastZ = z;
        lastX = input;

        for (int i = 0; i < inputLength; i++) {
            for (int j = 0; j < outputLength; j++) {
                out[j] = reLu(z[j]);
            }
        }

        return out;

    }

    @Override 
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedForwardPass(input);
        if (nextLayer != null) {
            return nextLayer.getOutput(forwardPass);
        } else {
            return forwardPass;
        }
    }

    @Override
    public void backPropagation(double[] dLdO) {
        
        double[] dLdX = new double[inputLength];
        
        double dOdZ;
        double dZdw;
        double dLdw;
        double dzdx;

        for(int k = 0; k < inputLength; k++){
            
            double dLdX_sum = 0;

            for(int j = 0; j < outputLength; j++){
                dOdZ = (lastZ[j] > 0) ? 1 : leak; 
                dZdw = lastX[k]; 
                dzdx = weights[k][j];

                dLdw = dLdO[j] * dOdZ * dZdw;

                weights[k][j] -= learningRate * dLdw;

                dLdX_sum += dLdO[j] * dOdZ * dzdx;
            }

            dLdX[k] = dLdX_sum;
        }
        if(previousLayer != null) {
            previousLayer.backPropagation(dLdX);
        }
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        double[] vector = matrixToVector(dLdO);
        backPropagation(vector);
    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
       return 0;
    }

    @Override
    public int getOutputCols() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return outputLength;
    }

    public void setRandomWeights() {
        Random rand = new Random(SEED);
        
        for (int i = 0; i < inputLength; i++) {
            for (int j = 0; j < outputLength; j++) {
                weights[i][j] = rand.nextGaussian();
            }
        }
    }


    public double reLu(double x) {
        return Math.max(0, x);
    }
}
