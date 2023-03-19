using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LibStableDiffusion;

internal class VaeDecoder
{
    internal static Image<Rgba32> ConvertToImage(Tensor<float> output, int width = 512, int height = 512)
    {
        var result = new Image<Rgba32>(width, height);

        for (var y = 0; y < height; y++)
            for (var x = 0; x < width; x++)
                result[x, y] = new Rgba32(
                    (byte)Math.Round(Math.Clamp((output[0, 0, y, x] / 2 + 0.5), 0, 1) * 255),
                    (byte)Math.Round(Math.Clamp((output[0, 1, y, x] / 2 + 0.5), 0, 1) * 255),
                    (byte)Math.Round(Math.Clamp((output[0, 2, y, x] / 2 + 0.5), 0, 1) * 255)
                );
        return result;
    }

    internal static Tensor<float> Decoder(List<NamedOnnxValue> input)
    {
        var vaeDecoderModelPath = Config.ReadSetting<string>("VaeDecoder");

        var sessionOptions = new SessionOptions();
        // Set CUDA EP
        if (Config.ReadSetting<string>("Provider") == "CUDA")
            sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider();

        // Create an InferenceSession from the Model Path.
        var vaeDecodeSession = new InferenceSession(vaeDecoderModelPath, sessionOptions);

        // Run session and send the input data in to get inference output. 
        var output = vaeDecodeSession.Run(input);
        var result = (Tensor<float>)output.ToList().First().Value;

        return result;
    }
}