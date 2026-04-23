using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace RealEstatePricePrediction
{
    public class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 42);

            string dataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "properties.csv");

            if (!File.Exists(dataPath))
            {
                Console.WriteLine($"Файлът не е намерен: {dataPath}");
                Console.WriteLine("Увери се, че properties.csv е в output папката на проекта.");
                return;
            }

            // Зареждане на CSV без атрибути
            var loader = mlContext.Data.CreateTextLoader(new TextLoader.Options
            {
                HasHeader = true,
                Separators = new[] { ',' },
                Columns = new[]
                {
                    new TextLoader.Column("Area", DataKind.Single, 0),
                    new TextLoader.Column("Rooms", DataKind.Single, 1),
                    new TextLoader.Column("Floor", DataKind.Single, 2),
                    new TextLoader.Column("YearBuilt", DataKind.Single, 3),
                    new TextLoader.Column("District", DataKind.String, 4),
                    new TextLoader.Column("NearMetro", DataKind.Single, 5),
                    new TextLoader.Column("BuildingType", DataKind.String, 6),
                    new TextLoader.Column("Price", DataKind.Single, 7)
                }
            });

            IDataView data = loader.Load(dataPath);

            // Разделяне на train/test
            var splitData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // Pipeline за обработка и обучение
            var pipeline =
                mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Price")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(
                    outputColumnName: "DistrictEncoded",
                    inputColumnName: "District"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(
                    outputColumnName: "BuildingTypeEncoded",
                    inputColumnName: "BuildingType"))
                .Append(mlContext.Transforms.Concatenate(
                    "Features",
                    "Area",
                    "Rooms",
                    "Floor",
                    "YearBuilt",
                    "NearMetro",
                    "DistrictEncoded",
                    "BuildingTypeEncoded"))
                .Append(mlContext.Regression.Trainers.FastTree(
                labelColumnName: "Label",
                featureColumnName: "Features"));

            Console.WriteLine("Обучение на модела...");
            Console.WriteLine("Започва Fit...");
            var model = pipeline.Fit(splitData.TrainSet);
            Console.WriteLine("Fit приключи.");

            // Оценка
            var predictions = model.Transform(splitData.TestSet);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine();
            Console.WriteLine("=== Метрики на модела ===");
            Console.WriteLine($"R²:   {metrics.RSquared:F4}");
            Console.WriteLine($"MAE:  {metrics.MeanAbsoluteError:F2}");
            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:F2}");

            // Прогноза за примерен имот
            var predictionEngine = mlContext.Model.CreatePredictionEngine<PropertyData, PropertyPrediction>(model);

            var sampleProperty = new PropertyData
            {
                Area = 85,
                Rooms = 3,
                Floor = 4,
                YearBuilt = 2016,
                District = "Mladost",
                NearMetro = 1,
                BuildingType = "Brick",
                Price = 0
            };

            var result = predictionEngine.Predict(sampleProperty);

            Console.WriteLine();
            Console.WriteLine("=== Примерна прогноза ===");
            Console.WriteLine($"Площ: {sampleProperty.Area} кв.м");
            Console.WriteLine($"Стаи: {sampleProperty.Rooms}");
            Console.WriteLine($"Етаж: {sampleProperty.Floor}");
            Console.WriteLine($"Година: {sampleProperty.YearBuilt}");
            Console.WriteLine($"Квартал: {sampleProperty.District}");
            Console.WriteLine($"Близо до метро: {(sampleProperty.NearMetro == 1 ? "Да" : "Не")}");
            Console.WriteLine($"Тип строителство: {sampleProperty.BuildingType}");
            Console.WriteLine($"Прогнозна цена: {result.Score:F2} EUR");

            Console.WriteLine();
            Console.WriteLine("Натисни произволен клавиш за край...");
            Console.ReadKey();
        }
    }

    public class PropertyData
    {
        public float Area { get; set; }
        public float Rooms { get; set; }
        public float Floor { get; set; }
        public float YearBuilt { get; set; }
        public string District { get; set; }
        public float NearMetro { get; set; }
        public string BuildingType { get; set; }
        public float Price { get; set; }
    }

    public class PropertyPrediction
    {
        public float Score { get; set; }
    }
}