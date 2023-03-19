using System.Reflection;
using System.Text.Json.Nodes;

namespace LibStableDiffusion;

internal class Config
{
    internal static T ReadSetting<T>(string key)
    {
        T result;
        using (var stream = new StreamReader(Path.GetDirectoryName(Assembly.GetAssembly(typeof(Config))!.Location) + "\\config.json"))
        {
            var json = JsonNode.Parse(stream.ReadToEnd());
            result = json![key]!.GetValue<T>();
        }
        return result;
    }
}
