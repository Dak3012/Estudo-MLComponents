using AI.Classes;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using Xunit;

namespace AITeste
{
    public class UnitTest1
    {
        private List<House> CreateRandosHouses(int number)
        {
            var houses = new List<House>();
            var rand = new Random();
            for (var x = 0; x < number; x++)
            {
                var size = (float)rand.Next(1, 10);
                var Comodos = rand.Next(1, 3);
                var price = (size * (3-Comodos));
                houses.Add(new House
                {
                    Size = size,
                    Comodos = Comodos,
                    Price = price,
                });
            }
            return houses;
        }
        private ITransformer CreateModelTeste()
        {
            var houses = CreateRandosHouses(1000);
            var AI = new AIClass(houses);
            return AI.Training();
        }
        [Fact]
        public void TestModelTeste()
        {
            var AI = new AIClass(CreateModelTeste());
            var houses = CreateRandosHouses(10000);
            var result = AI.TestModel(houses);
        }
        [Fact]
        public void PredictModelTeste()
        {
            var AI = new AIClass(CreateModelTeste());
            var house = new House
            {
                Size = 2,
                Comodos = 2,
            };
            var price = (house.Size * (3 - house.Comodos));
            var teste = AI.Predict(house);
        }
    }
}
