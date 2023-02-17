generating the dialogue answering structures.

```
python parsing.py \
--in_path xx\
--out_path xx\
--data_dir dataset\
--output_dir ./checkpoint/ \
--train_batch_size 2 \
--eval_batch_size 2 \
--do_train --do_test \
--do_lower_case \
--task_name bert_v7 \
--max_seq_length 128 \
--learning_rate 5e-6
```


