# Commands

## To filter questions
```
python select_gender_free_prompts.py --max-count 50
```

## To grade responses
```
python -m simple_evals.simple_evals --eval=healthbench_hard --model=qwen --n-threads 3
```
