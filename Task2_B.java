import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Task2_B {

    public static class KMeansMapper extends Mapper<LongWritable, Text, Text, Text> {

        private Text Keycentroid = new Text();
        private Text Value = new Text();
        private ArrayList<Convergence.Point<Integer, Integer>> seeds = new ArrayList<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            URI[] cacheFiles = context.getCacheFiles();
            Path path = new Path(cacheFiles[0]);

            // open the stream
            FileSystem fs = FileSystem.get(context.getConfiguration());
            FSDataInputStream fis = fs.open(path);

            // wrap it into a BufferedReader object which is easy to read a record
            BufferedReader reader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            String line;

            while ((line = reader.readLine()) != null) {
                String[] fields = line.split(",");
                Convergence.Point<Integer, Integer> seed = new Convergence.Point<>(Integer.parseInt(fields[0]), Integer.parseInt(fields[1]));
                seeds.add(seed);
            }
        }

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] point = value.toString().split(",");

            String closestCentroid = ClosestCentroid(point);

            Keycentroid.set(closestCentroid);
            Value.set(value);

            context.write(Keycentroid, Value);
        }

        private String ClosestCentroid(String[] point) {
            int x = Integer.parseInt(point[0]);
            int y = Integer.parseInt(point[1]);

            double currentMinDist = Double.MAX_VALUE;
            Convergence.Point<Integer, Integer> currentCentroid = null;

            for (Convergence.Point<Integer, Integer> seed : seeds) {
                double distance = Math.sqrt(Math.pow(x - seed.x, 2) + Math.pow(y - seed.y, 2));
                if (distance < currentMinDist) {
                    currentMinDist = distance;
                    currentCentroid = seed;
                }
            }
            return currentCentroid.x + "," + currentCentroid.y;
        }
    }

    public static class KMeansReducer extends Reducer<Text, Text, Text, NullWritable> {

        private Text newCentroid = new Text();

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            int sumX = 0;
            int sumY = 0;
            int count = 0;

            for (Text value : values) {
                String[] point = value.toString().split(",");
                sumX += Integer.parseInt(point[0].trim());
                sumY += Integer.parseInt(point[1].trim());
                count++;
            }
            int centroidX = sumX / count;
            int centroidY = sumY / count;
            newCentroid.set(centroidX + "," + centroidY);
            context.write(newCentroid, NullWritable.get());
        }
    }

    public static void main(String[] args) throws Exception {
        // start time
        long startTime = System.currentTimeMillis();

        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        String inputPath = "input/Dataset.csv"; // stay the same through all iterations
        String seedsPath = "input/seed_points_10.csv"; // changes. first is added to cache file but in the next iteration add the file with /iteration_ to the inputPath
        String outputPathBase = "MapReduceTasks/Part2_task_B";

        final int R = 30;
        for (int i = 0; i < R; i++) {
            Job job = Job.getInstance(conf, "KMeans Clustering - Iteration " + (i + 1));

            job.setJarByClass(Task2_B.class);

            // Set input path: For the first iteration, use the initial input. For subsequent iterations, use the output of the previous iteration
            FileInputFormat.addInputPath(job, new Path(inputPath));
            // Add the seeds/centroids file to the distributed cache for this job
            job.addCacheFile(new URI(i == 0 ? seedsPath : (outputPathBase + "/iteration_" + i + "/part-r-00000")));

            // Set output path for this iteration
            Path outputPath = new Path(outputPathBase + "/iteration_" + (i + 1));
            if (fs.exists(outputPath)) {
                fs.delete(outputPath, true);
            }
            FileOutputFormat.setOutputPath(job, outputPath);

            job.setMapperClass(KMeansMapper.class);
            job.setReducerClass(KMeansReducer.class);

            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);

            boolean success = job.waitForCompletion(true);
            if (!success) {
                System.out.println("Clustering failed on iteration " + (i + 1));
                System.exit(1);
            }
        }

        // End time and calculate total time
        long endTime = System.currentTimeMillis();
        long elapsedTime = endTime - startTime;

        System.out.println("Task2_BKMeans Clustering for " + R + " iterations in Task2_b completed after: " + elapsedTime + " miliseconds");

        // check for overall convergence
        boolean hasConverged = Convergence.checkConvergence(
                outputPathBase + "/iteration_" + (R-1) + "/part-r-00000",
                outputPathBase + "/iteration_" + R + "/part-r-00000",
                0.5
        );

        if (hasConverged){
            System.out.println("Converged");
        } else {
            System.out.println("Not Converged");
        }
    }
}
