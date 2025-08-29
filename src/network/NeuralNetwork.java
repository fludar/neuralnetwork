package network;

import java.util.ArrayList;
import java.util.List;

import data.Image;
import static data.MatrixUtility.*;
import layers.Layer;

public class NeuralNetwork {
    List<Layer> layers;
    double scaleFactor;

    public NeuralNetwork(List<Layer> layers, double scaleFactor) {
        this.layers = layers;
        this.scaleFactor = scaleFactor;
        linkLayers();
    }

    private void linkLayers() {
        if(layers.size() <= 1) return;

        for(int i = 0; i < layers.size(); i++) {
            if(i == 0){
                layers.get(i).setNextLayer(layers.get(i + 1));
            }
            else if(i == layers.size() - 1) {
                layers.get(i).setPreviousLayer(layers.get(i - 1));
            }
            else {
                layers.get(i).setNextLayer(layers.get(i + 1));
                layers.get(i).setPreviousLayer(layers.get(i - 1));
            }
        }
    }

    public double[] getErrors(double[] networkOutput, int correctAnswer) {
        int numClasses = networkOutput.length;

        double[] expectedOutput = new double[numClasses];
        
        expectedOutput[correctAnswer] = 1.0;
        
        return add(networkOutput, multiply(expectedOutput, -1.0));
    }

    private int getMaxIndex(double[] array) {
        double max = 0;
        int index = 0;
        
        for(int i = 0; i < array.length; i++) {
            if(array[i] >= max) {
                max = array[i];
                index = i;
            }
        }

        return index;
    }

    public int guess(Image image) {
        List<double[][]> inList = new ArrayList<>();
        inList.add(multiply(image.getData(), (1.0/scaleFactor)));
        
        double[] output = layers.get(0).getOutput(inList);
        int guess = getMaxIndex(output);

        return guess;
    }

    public float test (List<Image> images) {
        int correct = 0;
        for (Image img : images) {
            int guess = guess(img);
            if(guess == img.getLabel()) {
                    correct++;
            }
        }
        return ((float)correct / images.size());
    }

    public void train(List<Image> images) {
        for (Image img : images) {
            List<double[][]> inList = new ArrayList<>();
            inList.add(multiply(img.getData(), (1.0/scaleFactor)));

            double[] output = layers.get(0).getOutput(inList);
            double[] dLdO = getErrors(output, img.getLabel());

            layers.get(layers.size() - 1).backPropagation(dLdO);
        }
    }

}
