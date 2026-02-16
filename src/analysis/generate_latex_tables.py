
import os


# Data from literature and our experiments
models_data = [
    {"Model": "ResNet-20", "Type": "CNN", "Acc": "92.60\%", "Params": "0.27M", "FLOPs": "40.81M", "Ref": "\\cite{he2016deep}"},
    {"Model": "ResNet-32", "Type": "CNN", "Acc": "93.53\%", "Params": "0.47M", "FLOPs": "69.12M", "Ref": "\\cite{he2016deep}"},
    {"Model": "ResNet-44", "Type": "CNN", "Acc": "94.01\%", "Params": "0.66M", "FLOPs": "97.44M", "Ref": "\\cite{he2016deep}"},
    {"Model": "ResNet-56", "Type": "CNN", "Acc": "94.37\%", "Params": "0.86M", "FLOPs": "125.75M", "Ref": "\\cite{he2016deep}"},
    {"Model": "VGG-16 (bn)", "Type": "CNN", "Acc": "94.16\%", "Params": "15.25M", "FLOPs": "313.73M", "Ref": "\\cite{simonyan2014very}"},
    {"Model": "MobileNetV2 (0.5x)", "Type": "Efficient", "Acc": "92.88\%", "Params": "0.70M", "FLOPs": "27.97M", "Ref": "\\cite{sandler2018mobilenetv2}"},
    {"Model": "MobileNetV2 (1.0x)", "Type": "Efficient", "Acc": "93.79\%", "Params": "2.24M", "FLOPs": "87.98M", "Ref": "\\cite{sandler2018mobilenetv2}"},
    {"Model": "ShuffleNetV2 (0.5x)", "Type": "Efficient", "Acc": "90.13\%", "Params": "0.35M", "FLOPs": "10.90M", "Ref": "\\cite{ma2018shufflenet}"},
    {"Model": "RepVGG-A0", "Type": "Rep", "Acc": "94.39\%", "Params": "7.84M", "FLOPs": "489.08M", "Ref": "\\cite{ding2021repvgg}"},
    {"Model": "\\textbf{MobileNetECA-Rep (Ours)}", "Type": "\\textbf{Hybrid}", "Acc": "\\textbf{93.49\%}", "Params": "\\textbf{0.08M}", "FLOPs": "\\textbf{10.73M}", "Ref": "This Work"}
]

output_dir = "/workspace/tesi-laurea/reports/tables"
os.makedirs(output_dir, exist_ok=True)

latex_code = """
\\begin{table}[h]
    \\centering
    \\label{tab:comparison_sota}
    \\caption{Confronto con lo Stato dell'Arte su CIFAR-10. Il nostro modello (MobileNetECA-Rep) ottiene risultati competitivi con una frazione dei parametri e dei FLOPs.}
    \\begin{tabular}{l l c c c c}
        \\toprule
        \\textbf{Modello} & \\textbf{Tipo} & \\textbf{Accuratezza} & \\textbf{Parametri} & \\textbf{FLOPs} & \\textbf{Rif.} \\\\
        \\midrule
"""

for model in models_data:
    latex_code += f"        {model['Model']} & {model['Type']} & {model['Acc']} & {model['Params']} & {model['FLOPs']} & {model['Ref']} \\\\\n"

latex_code += """        \\bottomrule
    \\end{tabular}
\\end{table}
"""

output_path = os.path.join(output_dir, "sota_comparison.tex")
with open(output_path, "w") as f:
    f.write(latex_code)

print(f"LaTeX table generated at: {output_path}")
print(latex_code)
