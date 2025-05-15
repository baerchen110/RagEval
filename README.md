# Sample project to evaluate Elastic RAG options

fill the scripts/.env file

```console
cd scripts
pip install -r requirements.txt
``` 

run imports sample
```console
python ground_truth_generation/import.py
```
run different rag scenarios
```console
python rag_scenarios/retriever_xxx.py
```

run different evaluation scripts
```console
python evaluation/eval_xxx.py
```

run result printing  scripts
```console
python print_results/print_xxx.py
```

