package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.List;
import java.util.ArrayList;

public class DataReader {
    private final int rows = 28;
    private final int cols = 28;

    public List<Image> readData(String path) {
        
        List<Image> images = new ArrayList<>();
        
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            
            String line;
        
            while((line = br.readLine()) != null){
                String[] lineItems = line.split(",");

                double [][] pixels = new double[rows][cols];

                int label = Integer.parseInt(lineItems[0]);

                int i = 1;

                for(int row = 0; row < rows; row++){
                    for(int col = 0; col < cols; col++){
                        pixels[row][col] = (double) Integer.parseInt(lineItems[i]);
                        i++;
                    }
                }

                images.add(new Image(pixels, label));
            } 
        } catch (Exception e) {
            
        }

        return images;
    }
}
