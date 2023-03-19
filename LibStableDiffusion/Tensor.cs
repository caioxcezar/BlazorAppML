using Microsoft.ML.OnnxRuntime.Tensors;
using System.Management;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace LibStableDiffusion;
internal class Tensor
{
    internal static DenseTensor<T> Build<T>(T[] data, int[] dimensions) =>
        new DenseTensor<T>(data, dimensions);

    internal static DenseTensor<float> Duplicate(float[] data, int[] dimensions) =>
        Build(data.Concat(data).ToArray(), dimensions);

    internal static DenseTensor<float> DivideTensorByFloat(float[] data, float value, int[] dimensions) =>
        Build(data.Select((dt) => dt / value).ToArray(), dimensions);

    internal static Tensor<float> MultipleTensorByFloat(float[] data, float value, int[] dimensions) =>
        Build(data.Select((dt) => dt * value).ToArray(), dimensions);

    internal static DenseTensor<float> SumTensors(Tensor<float>[] tensorArray, int[] dimensions)
    {
        var sumTensor = new DenseTensor<float>(dimensions);
        var sumArray = new float[sumTensor.Length];

        for (int m = 0; m < tensorArray.Count(); m++)
        {
            var tensorToSum = tensorArray[m].ToArray();
            for (var i = 0; i < tensorToSum.Length; i++)
            {
                sumArray[i] += tensorToSum[i];
            }
        }

        return Build(sumArray, dimensions);
    }

    internal static DenseTensor<float> AddTensors(float[] sample, float[] sumTensor, int[] dimensions) =>
        Build(sample.Select((dt, i) => dt + sumTensor[i]).ToArray(), dimensions);

    internal static Tuple<Tensor<float>, Tensor<float>> SplitTensor(DenseTensor<float> tensorToSplit, int[] dimensions)
    {
        var tensor1 = new DenseTensor<float>(dimensions);
        var tensor2 = new DenseTensor<float>(dimensions);

        for (int i = 0; i < 1; i++)
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 512 / 8; k++)
                    for (int l = 0; l < 512 / 8; l++)
                    {
                        tensor1[i, j, k, l] = tensorToSplit[i, j, k, l];
                        tensor2[i, j, k, l] = tensorToSplit[i, j + 4, k, l];
                    }
        return new Tuple<Tensor<float>, Tensor<float>>(tensor1, tensor2);
    }

}