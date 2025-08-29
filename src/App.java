import data.DataReader;
import data.Image;
import network.NetworkBuilder;
import network.NeuralNetwork;

import java.util.Collections;
import java.util.List;

public class App {
    public static void main(String[] args) throws Exception {
        
        long SEED = 123;

        System.out.println("Begin data read");
        
        List<Image> imagesTest = new DataReader().readData("data/mnist_test.csv");
        List<Image> imagesTrain = new DataReader().readData("data/mnist_train.csv");
        
        System.out.println("Data read complete");

        System.out.println("Images test size: " + imagesTest.size());
        System.out.println("Images train size: " + imagesTrain.size());

        System.out.println("Begin building network");
        NetworkBuilder builder = new NetworkBuilder(28, 28, 256*100);
        System.out.println("Adding layers 1/3");
        builder.addConvolutionLayer(8, 5, 1, 0.1, SEED);
        System.out.println("Adding layers 2/3");
        builder.addMaxPoolLayer(3, 2);
        System.out.println("Adding layers 3/3");
        builder.addFullyConnectedLayer(10, 0.1, SEED);
        System.out.println("Done adding layers");

        NeuralNetwork network = builder.build();
        System.out.println("Network build complete");

        System.out.println("Begin training");
        float rate = network.test(imagesTest);
        System.out.println("Testing complete. Accuracy: " + rate);

        int epochs = 5;

        for(int i = 0; i < epochs; i++) {
            Collections.shuffle(imagesTrain);
            network.train(imagesTrain);
            rate = network.test(imagesTest);
            System.out.println("Success rate after round " + i + ": " + rate);
        }
    }
}
