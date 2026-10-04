# Fig. 7 reference evaluation outputs

These outputs were generated with the original sparse and dense checkpoints.
`fig7_abc.csv` and `fig7_d.csv` contain the evaluation curves; `run.json` records
the evaluation settings, and `checkpoints.json` records the weight checksums.

~~~bash
python scripts/plot_fig7.py --input reference_results --output results/reference_fig7
~~~
