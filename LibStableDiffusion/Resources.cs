using System.Reflection;

namespace LibStableDiffusion
{
    internal class Resources
    {
        internal static string Get(string resource)
        {
            var file = Path.GetDirectoryName(Assembly.GetAssembly(typeof(Resources))!.Location) + "\\Resources\\" + resource;
            if (File.Exists(file)) return file;
            throw new FileNotFoundException();
        }
    }
}