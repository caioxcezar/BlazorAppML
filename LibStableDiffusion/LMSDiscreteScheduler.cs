using MathNet.Numerics;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;

namespace LibStableDiffusion;

internal class LMSDiscreteScheduler
{
    private int NumTrainTimesteps;
    private string PredictionType;
    private List<float> AlphasCumulativeProducts;

    public Tensor<float> Sigmas { get; private set; }
    public List<int> TimeSteps { get; private set; }
    public List<Tensor<float>> Derivatives { get; private set; }
    public float InitNoiseSigma { get; private set; }

    public LMSDiscreteScheduler(int numInferenceSteps,
                                int numTrainTimesteps = 1000,
                                float betaStart = 0.00085f,
                                float betaEnd = 0.012f,
                                string betaSchedule = "scaled_linear",
                                string predictionType = "epsilon",
                                List<float>? trainedBetas = null)
    {
        NumTrainTimesteps = numTrainTimesteps;
        PredictionType = predictionType;
        AlphasCumulativeProducts = new List<float>();
        Derivatives = new List<Tensor<float>>();
        TimeSteps = new List<int>();

        var alphas = new List<float>();
        var betas = new List<float>();

        if (trainedBetas != null)
        {
            betas = trainedBetas;
        }
        else if (betaSchedule == "linear")
        {
            betas = Enumerable.Range(0, numTrainTimesteps).Select(i => betaStart + (betaEnd - betaStart) * i / (numTrainTimesteps - 1)).ToList();
        }
        else if (betaSchedule == "scaled_linear")
        {
            var start = (float)Math.Sqrt(betaStart);
            var end = (float)Math.Sqrt(betaEnd);
            betas = np.linspace(start, end, numTrainTimesteps).ToArray<float>().Select(x => x * x).ToList();

        }
        else
        {
            throw new Exception("beta_schedule must be one of 'linear' or 'scaled_linear'");
        }

        alphas = betas.Select(beta => 1 - beta).ToList();

        AlphasCumulativeProducts = alphas.Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b)).ToList();
        // Create sigmas as a list and reverse it
        var sigmas = AlphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();

        // standard deviation of the initial noise distrubution
        InitNoiseSigma = (float)sigmas.Max();

        Sigmas = new DenseTensor<float>(sigmas.Count());
        SetTimesteps(numInferenceSteps);
    }

    // Line 157 of scheduling_lms_discrete.py from HuggingFace diffusers
    private void SetTimesteps(int numInferenceSteps)
    {
        double start = 0;
        double stop = NumTrainTimesteps - 1;
        double[] timesteps = np.linspace(start, stop, numInferenceSteps).ToArray<double>();

        TimeSteps = timesteps.Select(x => (int)x).Reverse().ToList();

        var sigmas = AlphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();
        var range = np.arange(0d, (sigmas.Count)).ToArray<double>();
        sigmas = Interpolate(timesteps, range, sigmas).ToList();
        Sigmas = new DenseTensor<float>(sigmas.Count());
        for (int i = 0; i < sigmas.Count(); i++)
        {
            Sigmas[i] = (float)sigmas[i];
        }
    }

    public static double[] Interpolate(double[] timesteps, double[] range, List<double> sigmas)
    {

        // Create an output array with the same shape as timesteps
        var result = np.zeros(timesteps.Length + 1);

        // Loop over each element of timesteps
        for (int i = 0; i < timesteps.Length; i++)
        {
            // Find the index of the first element in range that is greater than or equal to timesteps[i]
            int index = Array.BinarySearch(range, timesteps[i]);

            // If timesteps[i] is exactly equal to an element in range, use the corresponding value in sigma
            if (index >= 0) result[i] = sigmas[index];
            // If timesteps[i] is less than the first element in range, use the first value in sigmas
            else if (index == -1) result[i] = sigmas[0];
            // If timesteps[i] is greater than the last element in range, use the last value in sigmas
            else if (index == -range.Length - 1) result[i] = sigmas[-1];
            // Otherwise, interpolate linearly between two adjacent values in sigmas
            else
            {
                index = ~index; // bitwise complement of j gives the insertion point of x[i]
                double t = (timesteps[i] - range[index - 1]) / (range[index] - range[index - 1]); // fractional distance between two points
                result[i] = sigmas[index - 1] + t * (sigmas[index] - sigmas[index - 1]); // linear interpolation formula
            }

        }
        //  add 0.000 to the end of the result
        result = np.add(result, 0.000f);

        return result.ToArray<double>();
    }

    public DenseTensor<float> ScaleInput(DenseTensor<float> sample, int timestep)
    {
        // Get step index of timestep from TimeSteps
        int stepIndex = this.TimeSteps.IndexOf(timestep);
        // Get sigma at stepIndex
        var sigma = this.Sigmas[stepIndex];
        sigma = (float)Math.Sqrt((Math.Pow(sigma, 2) + 1));

        // Divide sample tensor shape {2,4,64,64} by sigma
        sample = Tensor.DivideTensorByFloat(sample.ToArray(), sigma, sample.Dimensions.ToArray());

        return sample;
    }

    //python line 135 of scheduling_lms_discrete.py
    public double GetLmsCoefficient(int order, int t, int currentOrder)
    {
        // Compute a linear multistep coefficient.

        double LmsDerivative(double tau)
        {
            double prod = 1.0;
            for (int k = 0; k < order; k++)
            {
                if (currentOrder == k)
                {
                    continue;
                }
                prod *= (tau - this.Sigmas[t - k]) / (this.Sigmas[t - currentOrder] - this.Sigmas[t - k]);
            }
            return prod;
        }

        double integratedCoeff = Integrate.OnClosedInterval(LmsDerivative, this.Sigmas[t], this.Sigmas[t + 1], 1e-4);

        return integratedCoeff;
    }

    public DenseTensor<float> Step(
           Tensor<float> modelOutput,
           int timestep,
           Tensor<float> sample,
           int order = 4)
    {
        int stepIndex = TimeSteps.IndexOf(timestep);
        var sigma = Sigmas[stepIndex];

        // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        Tensor<float> predOriginalSample;

        // Create array of type float length modelOutput.length
        float[] predOriginalSampleArray = new float[modelOutput.Length];
        var modelOutPutArray = modelOutput.ToArray();
        var sampleArray = sample.ToArray();

        if (PredictionType == "epsilon")
        {

            for (int i = 0; i < modelOutPutArray.Length; i++)
            {
                predOriginalSampleArray[i] = sampleArray[i] - sigma * modelOutPutArray[i];
            }
            predOriginalSample = Tensor.Build(predOriginalSampleArray, modelOutput.Dimensions.ToArray());

        }
        else if (PredictionType == "v_prediction")
        {
            //predOriginalSample = modelOutput * ((-sigma / Math.Sqrt((Math.Pow(sigma,2) + 1))) + (sample / (Math.Pow(sigma,2) + 1)));
            throw new Exception($"prediction_type given as {PredictionType} not implemented yet.");
        }
        else
        {
            throw new Exception($"prediction_type given as {PredictionType} must be one of `epsilon`, or `v_prediction`");
        }

        // 2. Convert to an ODE derivative
        var derivativeItems = new DenseTensor<float>(sample.Dimensions.ToArray());

        var derivativeItemsArray = new float[derivativeItems.Length];

        for (int i = 0; i < modelOutPutArray.Length; i++)
        {
            //predOriginalSample = (sample - predOriginalSample) / sigma;
            derivativeItemsArray[i] = (sampleArray[i] - predOriginalSampleArray[i]) / sigma;
        }
        derivativeItems = Tensor.Build(derivativeItemsArray, derivativeItems.Dimensions.ToArray());

        Derivatives.Add(derivativeItems);

        if (Derivatives.Count() > order)
        {
            // remove first element
            Derivatives.RemoveAt(0);
        }

        // 3. compute linear multistep coefficients
        order = Math.Min(stepIndex + 1, order);
        var lmsCoeffs = Enumerable.Range(0, order).Select(currOrder => GetLmsCoefficient(order, stepIndex, currOrder)).ToArray();

        // 4. compute previous sample based on the derivative path
        // Reverse list of tensors this.derivatives
        var revDerivatives = Enumerable.Reverse(Derivatives).ToList();

        // Create list of tuples from the lmsCoeffs and reversed derivatives
        var lmsCoeffsAndDerivatives = lmsCoeffs.Zip(revDerivatives, (lmsCoeff, derivative) => (lmsCoeff, derivative));

        // Create tensor for product of lmscoeffs and derivatives
        var lmsDerProduct = new Tensor<float>[Derivatives.Count()];

        for (int m = 0; m < lmsCoeffsAndDerivatives.Count(); m++)
        {
            var item = lmsCoeffsAndDerivatives.ElementAt(m);
            // Multiply to coeff by each derivatives to create the new tensors
            lmsDerProduct[m] = Tensor.MultipleTensorByFloat(item.derivative.ToArray(), (float)item.lmsCoeff, item.derivative.Dimensions.ToArray());
        }
        // Sum the tensors
        var sumTensor = Tensor.SumTensors(lmsDerProduct, new[] { 1, 4, 64, 64 });

        // Add the sumed tensor to the sample
        var prevSample = Tensor.AddTensors(sample.ToArray(), sumTensor.ToArray(), sample.Dimensions.ToArray());

        return prevSample;
    }
}