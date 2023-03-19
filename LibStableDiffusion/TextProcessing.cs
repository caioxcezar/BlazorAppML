using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LibStableDiffusion;
internal class TextProcessing
{
    private static readonly int modelLength = 77;
    private static readonly int blankTokenValue = 49407;
    internal static int[] TokenizeText(string text)
    {
        // Create Tokenizer and tokenize the sentence.

        var tokenizerOnnxPath = Resources.Get("\\text_tokenizer\\custom_op_cliptok.onnx");

        // Create session options for custom op of extensions
        var sessionOptions = new SessionOptions();
        var customOp = Resources.Get("ortextensions.dll");
        sessionOptions.RegisterCustomOpLibraryV2(customOp, out var libraryHandle);

        // Create an InferenceSession from the onnx clip tokenizer.
        var tokenizeSession = new InferenceSession(tokenizerOnnxPath, sessionOptions);
        var inputTensor = Tensor.Build(new string[] { text }, new int[] { 1 });
        var inputString = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("string_input", inputTensor) };
        // Run session and send the input data in to get inference output. 
        var tokens = tokenizeSession.Run(inputString);

        var ids = new int[77];
        var inputIds = ((IEnumerable<long>)tokens.ToList().First().Value).Select(x => (int)x).ToArray();
        inputIds.CopyTo(ids, 0);

        // Pad array with 49407 until length is modelMaxLength
        for (var i = inputIds.Length; i < modelLength; i++)
            ids[i] = blankTokenValue;

        return ids;
    }

    public static int[] CreateUnconditionalInput()
    {
        // Create an array of empty tokens for the unconditional input. 
        var inputIds = new int[modelLength];
        inputIds[0] = 49406;
        for (int i = 1; i < modelLength; i++)
            inputIds[i] = blankTokenValue;
        return inputIds;
    }

    internal static DenseTensor<float> TextEncoder(string text) =>
        TextEncoder(TokenizeText(text));

    internal static DenseTensor<float> TextEncoder(int[] tokenizedInput)
    {
        // Create input tensor.
        var input_ids = Tensor.Build(tokenizedInput, new[] { 1, tokenizedInput.Count() });

        var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<int>("input_ids", input_ids) };

        var sessionOptions = new SessionOptions();
        // Set CUDA EP
        if (Config.ReadSetting<string>("Provider") == "CUDA")
            sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider();
        else
            sessionOptions.AppendExecutionProvider_CPU();

        var textEncoderOnnxPath = Config.ReadSetting<string>("TextEncoder");

        var encodeSession = new InferenceSession(textEncoderOnnxPath, sessionOptions);
        // Run inference.
        var encoded = encodeSession.Run(input);

        var lastHiddenState = ((IEnumerable<float>)encoded.ToList().First().Value).ToArray();
        var lastHiddenStateTensor = Tensor.Build(lastHiddenState.ToArray(), new[] { 1, 77, 768 });

        return lastHiddenStateTensor;
    }
}