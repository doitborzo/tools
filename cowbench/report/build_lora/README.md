# LoRA results report (Ukrainian)

Rebuilds `../Muse_Glimmer_LoRA_results_uk.docx` from `data.json`, which holds the
numbers of the five runs' `summary.md` tables (cowbench/summary.py), the
lora4frame reports and its stress tests. No number is typed into the text.

```bash
npm install docx            # once, in this folder
python chart.py             # chart_exact.png
node build.js ../Muse_Glimmer_LoRA_results_uk.docx
```
