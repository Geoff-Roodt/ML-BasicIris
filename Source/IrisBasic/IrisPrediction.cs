using Microsoft.ML.Runtime.Api;

namespace IrisBasic
{
    /// <summary>
    /// Iris Prediction is the result returned from prediction operations
    /// </summary>
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}
