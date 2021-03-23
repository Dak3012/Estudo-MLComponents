using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AI.Classes
{
    public class AIClass
    {
        private ITransformer Context;
        public ITransformer Training()
        {
            return Context;
        }
        public RegressionMetrics TestModel(List<House> Data)
        {
            var context = new MLContext();
            var TestData = context.Data.LoadFromEnumerable(Data);
            var test = Context.Transform(TestData);
            var cds = context.Regression.CrossValidate(test,context.Regression.Trainers.Sdca(labelColumnName: "Price"), numberOfFolds: 5, labelColumnName: "Price"); //fazer validação cruzada a parte
            var testModel = context.Regression.Evaluate(test, labelColumnName: "Price");
            return testModel;
        }
        public AIClass(ITransformer model)
        {
            Context = model;
        }
        public AIClass(List<House> Data)
        {
            var context = new MLContext();
            var TrainingData = context.Data.LoadFromEnumerable(Data);
            var sdaEstimator = context.Regression.Trainers.Sdca(labelColumnName: "Price");
            var pipeline = context.Transforms.Concatenate("Features", new[] { "Size", "Comodos" }).Append(sdaEstimator).Append(context.Transforms.NormalizeMinMax("Features")); 
            var model = pipeline.Fit(TrainingData);
            Context = model;
        }
        public Prediction Predict(House house)
        {
            var context = new MLContext();
            var price = context.Model.CreatePredictionEngine<House, Prediction>(Context).Predict(house);
            return price;
        }
    }
}
