## specifications

 - tools/utils.py: all help functions
 - optimize.py: the prompt with the fixed place holder required and light dimension requirement.
 - main.py: iterate evaluate and optimize
 - eval_val.py: run target val eval first; then select best topk by select_final.py; then eval the bench val, target test, bench test on these finals to speed up.
 
the executaion step can be referred from `./safety_task_double/main_double.sh`:

1. run main to optimize
2. select best topk from each generation by `select_rubrics.py`
3. eval all selected on target val
4. select another top candidates from target val, to final, by `select_final.py`
5. eval final ones on val bench to select the final, and also on bench, target test to see the results.


## TODO

`sh ./safety_task_double/main_double.sh`
