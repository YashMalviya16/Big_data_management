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

public class Task2_E {

    public static class KMeansMapper extends Mapper<LongWritable, Text, Text, Text> {

        private Text Keycentroid = new Text();
        private Text Value = new Text();
        private ArrayList<Convergence.Point<Integer, Integer>> seeds = new ArrayList<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {


            URI[] cacheFiles = context.getCacheFiles();
            Path path = new Path(cacheFiles[0]);
            FileSystem fs = FileSystem.get(context.getConfiguration());
            FSDataInputStream fis = fs.open(path);


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


    public static class KMeansCombiner extends Reducer<Text, Text, Text, Text> {
        private Text Sum = new Text();

        @Override
        protected void reduce (Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            int sumX = 0;
            int sumY = 0;
            int count = 0;

            for (Text value: values){
                String[] point = value.toString().split(",");
                sumX += Integer.parseInt(point[0].trim());
                sumY += Integer.parseInt(point[1].trim());
                count++;
            }

            Sum.set(sumX + "," + sumY + "," + count);
            context.write(key, Sum);
        }
    }

    public static class KMeansReducer extends Reducer<Text, Text, Text, NullWritable> {

        private Text newCentroid = new Text();
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            int totalX = 0;
            int totalY = 0;
            int Count = 0;

            for (Text value : values) {
                String[] DataCombined = value.toString().split(",");
                totalX += Integer.parseInt(DataCombined[0].trim());
                totalY += Integer.parseInt(DataCombined[1].trim());
                Count += Integer.parseInt(DataCombined[2].trim());
            }

            int centroidX = totalX / Count;
            int centroidY = totalY / Count;
            newCentroid.set(centroidX + "," + centroidY);

            context.write(newCentroid, NullWritable.get());
        }
    }

    public static void main(String[] args) throws Exception {
        long startTime = System.currentTimeMillis();
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        String inputPath = "input/Dataset.csv";
        String seedsPath = "input/seed_points_10.csv";
        String outputPathBase = "MapReduceTasks/Part2_task_E";
        boolean hasConverged = false;
        int maxIterations = 30;
        int IterationIndex = 0;
        int i;
        for ( i=0; i < maxIterations; i++) {
            Job job = Job.getInstance(conf, "Iteration " + (i + 1));
            job.addCacheFile(new URI(i == 0 ? seedsPath : (outputPathBase + "/iteration_" + i + "/part-r-00000")));
            job.setJarByClass(Task2_E.class);
            FileInputFormat.addInputPath(job, new Path(inputPath));
            Path outputPath = new Path(outputPathBase + "/iteration_" + (i + 1));
            if (fs.exists(outputPath)) {
                fs.delete(outputPath, true);
            }
            FileOutputFormat.setOutputPath(job, outputPath);

            job.setMapperClass(KMeansMapper.class);
            job.setCombinerClass(KMeansCombiner.class);
            job.setReducerClass(KMeansReducer.class);

            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);


            boolean success = job.waitForCompletion(true);
            if (!success) {
                System.out.println("failed on iteration " + (i + 1));
                System.exit(1);
            }


            if (i > 0) {
                hasConverged = Convergence.checkConvergence(
                        outputPathBase + "/iteration_" + i + "/part-r-00000",
                        outputPathBase + "/iteration_" + (i + 1) + "/part-r-00000",
                        0.5
                );
                if (hasConverged) {
                    IterationIndex = i + 1;
                    System.out.println("converged after " + IterationIndex + " iterations.");
                    break;
                }
            }
        }

        if (!hasConverged){
            IterationIndex = maxIterations;
        }
        Job clusterMappingJob = Job.getInstance(conf, "Clusters from Data ");

        clusterMappingJob.addCacheFile(new URI(outputPathBase + "/iteration_" + IterationIndex + "/part-r-00000")); // final centroids as cache

        clusterMappingJob.setJarByClass(Task2_E.class);

        FileInputFormat.addInputPath(clusterMappingJob, new Path(inputPath));

        Path finalOutputPath = new Path(outputPathBase + "/final_clusters");
        if (fs.exists(finalOutputPath)) {
            fs.delete(finalOutputPath, true);
        }
        FileOutputFormat.setOutputPath(clusterMappingJob, finalOutputPath);

        clusterMappingJob.setMapperClass(KMeansMapper.class);
        clusterMappingJob.setReducerClass(Reducer.class);
        clusterMappingJob.setOutputKeyClass(Text.class);
        clusterMappingJob.setOutputValueClass(Text.class);

        clusterMappingJob.waitForCompletion(true);



        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("Total runtime: " + totalTime + " milliseconds");

        if(hasConverged) {
            System.out.println(" converged after " + IterationIndex + " iterations.");
        } else {
            System.out.println("NOT converged after " + maxIterations + " iterations.");
        }
    }
}
