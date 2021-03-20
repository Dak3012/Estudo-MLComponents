using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AI.Classes
{
    public class AIClass
    {
        public void Training(List<House> Data)
        {
            var context = new MLContext();
            var TrainingData = context.Data.LoadFromEnumerable(Data);
            var pipeline = context.Transforms.Concatenate("Price", new[] { "Size", "Comodos" }).Append(context.Regression.Trainers.Sdca());
            var model = pipeline.Fit(TrainingData);
        }
        public void TestModel(List<Object> model)
        {

        }
        public void Load(Object model)
        {

        }
        public void Predict(Object Data)
        {

        }
    }
}
