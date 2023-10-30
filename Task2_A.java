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


public class Task2_A {

    public static class KMeansMapper extends Mapper<LongWritable, Text, Text, Text> {

        private Text Keycentroid = new Text();
        private Text Value = new Text();
        private ArrayList<Convergence.Point<Integer, Integer>> seedpoi = new ArrayList<>();

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
                seedpoi.add(seed);
            }
        }

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] point = value.toString().split(",");

            String closestCentroid = findClosestCentroid(point);

            Keycentroid.set(closestCentroid);
            Value.set(value);

            context.write(Keycentroid, Value);
        }

        private String findClosestCentroid(String[] point) {
            int x = Integer.parseInt(point[0]);
            int y = Integer.parseInt(point[1]);

            double currentMinDist = Double.MAX_VALUE;
            Convergence.Point<Integer, Integer> currentCentroid = null;

            for (Convergence.Point<Integer, Integer> seed : seedpoi) {
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

        Configuration conf = new Configuration();

        // clean up output directories if they exist currently
        FileSystem fs = FileSystem.get(conf);

        String inputPath = "input/Dataset.csv";
        String seedsPath = "input/seedpoints_5.csv";
        String outputPath = "MapReduceTasks/Part2_task_A";

        Path outPath = new Path(outputPath); // output path
        if (fs.exists(outPath)) {
            fs.delete(outPath, true);
        }
        Job job = Job.getInstance(conf, "KMeans Clustering");
        Path inputAllDataPath = new Path(inputPath);

        job.addCacheFile(new URI(seedsPath)); // path to the initial seed point files
        job.setJarByClass(Task2_A.class);

        FileInputFormat.addInputPath(job, inputAllDataPath); // path to the full data set
        FileOutputFormat.setOutputPath(job, outPath);

        job.setMapperClass(KMeansMapper.class);
        job.setReducerClass(KMeansReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        long startTime = System.currentTimeMillis();
        boolean success = job.waitForCompletion(true);
        long endTime = System.currentTimeMillis();
        long elapsedTime = endTime - startTime;

        System.out.println("Task2_A - Single K-Means iteration: " + elapsedTime + " milliseconds.");

        // check for convergence by comparing the initial seed points with the updated centroids
        boolean hasConverged = Convergence.checkConvergence(
                seedsPath,
                outputPath + "/part-r-00000",
                0.5
        );

        if (hasConverged){
            System.out.println("Converged");
        } else {
            System.out.println("Not Converged");
        }
        System.exit(success ? 0 : 1);
    }
}