using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

namespace AI.Classes
{
    public class AiSentimentalsClassBinary
    {
        public PredictionEngine<SentimentData, SentimentPrediction> Prediction { get; private set; }
        public CalibratedBinaryClassificationMetrics metricas { get; private set; }
        private string DataPath;
        private MLContext Context;
        private ITransformer model;

        public AiSentimentalsClassBinary(string Path)
        {
            DataPath = Path;
            Context = new MLContext();
            var arquivo = LoadData();
            model = BuildAndTrainModel(arquivo.TrainSet);
            metricas = Evaluate(arquivo.TestSet);
            
        }
        public AiSentimentalsClassBinary(ITransformer model)
        {
            model = this.model;
            Context = new MLContext();
            Prediction = Context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
        }
        public TrainTestData LoadData()
        {
            var DataView = Context.Data.LoadFromTextFile<SentimentData>(DataPath, hasHeader: false);
            var SplitDataView = Context.Data.TrainTestSplit(DataView, testFraction: 0.2);            
            return SplitDataView;
        }
        public ITransformer BuildAndTrainModel(IDataView SplitTrainTest)
        {
            var textFeaturizing = Context.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText));
            var sdcalogistic = Context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            var estimator = textFeaturizing.Append(sdcalogistic);
            var model = estimator.Fit(SplitTrainTest);
            Prediction = Context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            return model;
        }
        public CalibratedBinaryClassificationMetrics Evaluate(IDataView splitTest)
        {
            var prediction = model.Transform(splitTest);
            var metrics = Context.BinaryClassification.Evaluate(prediction, "Label");
            return metrics;
        }
        public PredictionEngine<SentimentData, SentimentPrediction> CarregarModelo(ITransformer model) //Utilizar para carregar um modelo
        {
            var predictionFunction = Context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            return predictionFunction;
        }
        public IEnumerable<SentimentPrediction> UseModelWithBatchItems(IEnumerable<SentimentData> sentiments)
        {
            var bathcomments = Context.Data.LoadFromEnumerable(sentiments);
            var preditictions = model.Transform(bathcomments);
            return Context.Data.CreateEnumerable<SentimentPrediction>(preditictions, reuseRowObject: false);
        }

    }
    public class SentimentData
    {
        [LoadColumn(0)]
        public string SentimentText;
        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }
    public class SentimentPrediction : SentimentData
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }
}
