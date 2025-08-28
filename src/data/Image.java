package data;

public class Image {
    private double[][] pixels;
    private int label;


    public double [][] getPixels() {
        return pixels;
    }

    public int getLabel() {
        return label;
    }

    public Image(double[][] pixels, int label) {
        this.pixels = pixels;
        this.label = label;
    }

    @Override
    public String toString() {

        String s = label + "\n";
        
        for(int i = 0; i < pixels.length; i++){
            for(int j = 0; j < pixels[0].length; j++){
                s += pixels[i][j] + ", ";
            }
            s += "\n";
        }


        return s;
    }

}
