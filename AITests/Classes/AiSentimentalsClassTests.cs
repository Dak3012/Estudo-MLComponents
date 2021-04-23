using Microsoft.VisualStudio.TestTools.UnitTesting;
using AI.Classes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace AI.Classes.Tests
{
    [TestClass()]
    public class AiSentimentalsClassTests
    {
        [TestMethod()]
        public void BuildAndTrainModelTest()
        {
            var DirStringEnvironment = Environment.GetEnvironmentVariable("Location");
            var path = Path.Combine(DirStringEnvironment, "yelp_labelled.txt");
            var createContext = new AiSentimentalsClassBinary(path);
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This was a very bad steak"
            };
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti."
                }
            };
            var predict = createContext.Prediction.Predict(sampleStatement);
            var lists2 = sentiments.Select(x => createContext.Prediction.Predict(x));
            var lists = createContext.UseModelWithBatchItems(sentiments).ToList(); //modelo apresentou lentidão em relação ao predict com lambda.
            
        }
    }
}