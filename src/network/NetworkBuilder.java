package network;

import java.util.ArrayList;
import java.util.List;
import layers.Layer;
import layers.MaxPoolLayer;
import layers.ConvolutionLayer;
import layers.FullyConnectedLayer;

public class NetworkBuilder {
    private NeuralNetwork network;

    private int inputRows;
    private int inputCols;
    private double scaleFactor;
    List<Layer> layers;

    public NetworkBuilder(int inputRows, int inputCols, double scaleFactor) {
        this.inputRows = inputRows;
        this.inputCols = inputCols;
        this.scaleFactor = scaleFactor;
        layers = new ArrayList<>();
    }

    public void addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate, long SEED) {
        if(layers.isEmpty()){
            layers.add(new ConvolutionLayer(filterSize, stepSize, 1, inputRows, inputCols, SEED, numFilters, learningRate));
        } else {
            Layer prev = layers.get(layers.size() - 1);
            layers.add(new ConvolutionLayer(filterSize, stepSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), SEED, numFilters, learningRate));
        }
    }

    public void addMaxPoolLayer(int windowSize, int stepSize){
        if(layers.isEmpty()){
            layers.add(new MaxPoolLayer(stepSize, windowSize, 1, inputRows, inputCols));
        } else {
            Layer prev = layers.get(layers.size()-1);
            layers.add(new MaxPoolLayer(stepSize, windowSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));
        }
    }

    public void addFullyConnectedLayer(int outLength, double learningRate, long SEED){
        if(layers.isEmpty()) {
            layers.add(new FullyConnectedLayer(inputCols * inputRows, outLength, SEED, learningRate));
        } else {
            Layer prev = layers.get(layers.size()-1);
            layers.add(new FullyConnectedLayer(prev.getOutputElements(), outLength, SEED, learningRate));
        }

    }

    public NeuralNetwork build() {
        network = new NeuralNetwork(layers, scaleFactor);
        return network;
    }
}
