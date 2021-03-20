using AI.Classes;
using System;
using System.Collections.Generic;
using Xunit;

namespace AITeste
{
    public class UnitTest1
    {
        [Fact]
        public void Test1()
        {
            var houses = new List<House>();
            var rand = new Random();
            for(var x = 0; x <= 100; x++)
            {
                var size = (float)rand.Next(1, 100) / 100;
                var Comodos = rand.Next(1, 3);
                var price = size * Comodos;
                houses.Add(new House
                {
                    Size= size,
                    Comodos = Comodos,
                    Price= price,
                });
            }
            var AI = new AIClass();

        }
    }
}
