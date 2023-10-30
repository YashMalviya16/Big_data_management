import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class Convergence {
    public static class Point<X, Y> {
        public final X x;
        public final Y y;
        public Point(X x, Y y) {
            this.x = x;
            this.y = y;
        }
    }

    public static boolean checkConvergence(String previousCentroidBasePath, String currentCentroidBasePath, double threshold) throws IOException {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path previousCentroidPath = new Path(previousCentroidBasePath);
        Path currentCentroidPath = new Path(currentCentroidBasePath);

        // Read the centroids from the previous iteration
        BufferedReader reader1 = new BufferedReader((new InputStreamReader(fs.open(previousCentroidPath), "UTF-8")));
        String line1;
        ArrayList<Point<Integer, Integer>> previousCentroids = new ArrayList<>();
        while ((line1 = reader1.readLine()) != null) {
            String[] centroidCoordinates = line1.split(",");
            int centroidX = Integer.parseInt(centroidCoordinates[0]);
            int centroidY = Integer.parseInt(centroidCoordinates[1]);

            previousCentroids.add(new Point<>(centroidX, centroidY));
        }
        reader1.close();

        // Read the centroids from the current iteration
        BufferedReader reader2 = new BufferedReader((new InputStreamReader(fs.open(currentCentroidPath), "UTF-8")));
        String line2;
        ArrayList<Point<Integer, Integer>> currentCentroids = new ArrayList<>();
        while ((line2 = reader2.readLine()) != null) {
            String[] centroidCoordinates = line2.split(",");
            int centroidX = Integer.parseInt(centroidCoordinates[0]);
            int centroidY = Integer.parseInt(centroidCoordinates[1]);

            currentCentroids.add(new Point<>(centroidX, centroidY));
        }
        reader2.close();

        // O(k*k) complexity but hopefully k is small
        for (Point<Integer, Integer> currentCentroid : currentCentroids) {
            double minDistance = Double.MAX_VALUE;
            for (Point<Integer, Integer> previousCentroid : previousCentroids) {
                double distance = Math.sqrt(Math.pow(currentCentroid.x - previousCentroid.x, 2) + Math.pow(currentCentroid.y - previousCentroid.y, 2));
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
            if (minDistance > threshold) {
                return false;
            }
        }
        return true;
    }
}
