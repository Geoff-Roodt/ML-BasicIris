using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;

namespace IrisBasic
{
    class Program
    {
        static void Main(string[] args)
        {
            // STEP 2: Create a pipeline and load the data from our flat file
            // The data loaded will be used to train the model
            LearningPipeline pipeline = new LearningPipeline();
            string dataPath = "iris-data.txt";
            pipeline.Add(new TextLoader(dataPath).CreateFrom<IrisData>(separator: ','));

            // STEP 3: Transform the data
            // We need to assign numeric values instead of the text in our 'Label' column;
            // strings can't be processed during model training
            pipeline.Add(new Dictionarizer("Label"));

            // Puts all features the model will train on into a vector
            pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            // STEP 4: Add Learner
            // Add a learning algorithm to the pipeline
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            // We should convert the label back into the original text that it was at before we numberized it!
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // STEP 5: Train thy model!
            PredictionModel<IrisData, IrisPrediction> model = pipeline.Train<IrisData, IrisPrediction>();

            // STEP 6: Get user input for model prediction
            float sepalLength = 0;
            float sepalWidth = 0;
            float petalLength = 0;
            float petalWidth = 0;

            while(sepalLength == 0)
            {
                Console.WriteLine("\r\nEnter the Sepal Length:");
                string length = Console.ReadLine();
                if (!float.TryParse(length, out sepalLength)){
                    Console.WriteLine("Sepal Length must be a float");
                }
            }

            while (sepalWidth == 0)
            {
                Console.WriteLine("\r\nEnter the Sepal Width:");
                string width = Console.ReadLine();
                if (!float.TryParse(width, out sepalWidth))
                {
                    Console.WriteLine("Sepal Width must be a float");
                }
            }

            while (petalLength == 0)
            {
                Console.WriteLine("\r\nEnter the Petal Length:");
                string length = Console.ReadLine();
                if (!float.TryParse(length, out petalLength))
                {
                    Console.WriteLine("Petal Length must be a float");
                }
            }

            while (petalWidth == 0)
            {
                Console.WriteLine("\r\nEnter the Petal Width:");
                string width = Console.ReadLine();
                if (!float.TryParse(width, out petalWidth))
                {
                    Console.WriteLine("Petal Width must be a float");
                }
            }
            
            //STEP 7: Predict based off the model
            var prediction = model.Predict(new IrisData()
            {
                SepalLength = sepalLength,
                SepalWidth = sepalWidth,
                PetalLength = petalLength,
                PetalWidth = petalWidth
            });


            Console.WriteLine($"Predicted flower that is: {prediction.PredictedLabels}");
            Console.ReadLine();
        }
    }
}
