using Microsoft.ML.OnnxRuntime.Tensors;
using System.Numerics;

namespace LibStableDiffusion;

public class Prompt
{
    public string Text { get; set; }
    /// <summary>
    /// Number of denoising steps
    /// </summary>
    public int? Steps { get; set; }
    /// <summary>
    /// Scale for classifier-free guidance
    /// </summary>
    public float? GuidanceScale { get; set; }
    /// <summary>
    /// Number of images requested
    /// </summary>
    public int? BatchSize { get; set; }
    /// <summary>
    /// Size of the image
    /// </summary>
    public Vector2 Size = new Vector2() { X = 512, Y = 512 };
}
public class StableDiffusion
{

    public static Image? GenerateImage(Prompt prompt)
    {
        var steps = prompt.Steps ?? 15;
        var scale = prompt.GuidanceScale ?? 7.5;
        var batchSize = prompt.BatchSize ?? 1;

        // Load the tokenizer and text encoder to tokenize and encode the text.
        var textPromptEmbeddings = TextProcessing.TextEncoder(prompt.Text).ToArray();

        // Create uncond_input of blank tokens
        var uncondInputTokens = TextProcessing.CreateUnconditionalInput();
        var uncondEmbedding = TextProcessing.TextEncoder(uncondInputTokens).ToArray();

        // Concant textEmeddings and uncondEmbedding
        var textEmbeddings = new DenseTensor<float>(new[] { 2, 77, 768 });

        for (var i = 0; i < textPromptEmbeddings.Length; i++)
        {
            textEmbeddings[0, i / 768, i % 768] = uncondEmbedding[i];
            textEmbeddings[1, i / 768, i % 768] = textPromptEmbeddings[i];
        }

        // Inference Stable Diff
        return UNet.Inference(steps, textEmbeddings, scale, batchSize, (int)prompt.Size.Y, (int)prompt.Size.X);
    }
}