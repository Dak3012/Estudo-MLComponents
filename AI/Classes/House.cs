using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AI.Classes
{
    public class House
    {
        public float Size { get; set; }
        public float Price { get; set; }
        public float Comodos { get; set; }
    }
    public class Prediction
    {
        [ColumnName("Score")]
        public float Price { get; set; }
    }
}
