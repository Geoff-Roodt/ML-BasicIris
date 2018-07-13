using Microsoft.ML.Runtime.Api;

namespace IrisBasic
{
    /// <summary>
    /// IrisData is used to provide training data, and as input for prediction operations
    /// - First 4 properties are inputs/features used to predict the label
    /// - Label is what you are predicting, and is only set when training the prediction model
    /// </summary>
    public class IrisData
    {
        [Column("0")]
        public float SepalLength;

        [Column("1")]
        public float SepalWidth;

        [Column("2")]
        public float PetalLength;

        [Column("3")]
        public float PetalWidth;

        [Column("4")]
        [ColumnName("Label")]
        public string Label;
    }
}
