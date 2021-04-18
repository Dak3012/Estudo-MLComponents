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
    public class AiSentimentalsClass
    {
        private string DataPath;
        private MLContext Context;
        public AiSentimentalsClass(string Path)
        {
            DataPath = Path;
            Context = new MLContext();
        }
        public TrainTestData LoadData()
        {
            var DataView = Context.Data.LoadFromTextFile<SentimentData>(DataPath, hasHeader: false);
            var SplitDataView = Context.Data.TrainTestSplit(DataView, testFraction:0.2);
            return SplitDataView;
        }
        public ITransformer BuildAndTrainModel(IDataView SplitTrainTest)
        {
            var textFeaturizing = Context.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText));
            var sdcalogistic = Context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            var estimator = textFeaturizing.Append(sdcalogistic);
            var model = estimator.Fit(SplitTrainTest);
            return model;
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
