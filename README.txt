# Easy-21

## Report File

There are three vesions report file: Report.md, Report.pdf and Report.html.

It is recommended to read the HTML version.

## Q-Learning

Train

```bash
cd ./QLearning
python Qlearning.py --no 300 --alpha 0.5 --variable --store_log --clear_dict --save_dict
```

Test

``` bash
cd ./QLearning
python Qlearning.py --test
```



## Policy Iteration

Train

You may need to delete the storage file ./Policy/Pi.dict and ./Policy/V.dict

```bash
cd ./Policy
python policy.py
```

Test

You may remove the comment on the last line and run ./Policy/policy.py