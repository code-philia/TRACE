lang=all
lr=5e-5
batch_size=4
source_length=512
data_dir=dataset_fine_grain
label_num=6
output_dir=model_${label_num}/$lang
train_file=$data_dir/$lang/train.json
dev_file=$data_dir/$lang/dev.json
test_file=$data_dir/$lang/test.json
epochs=2
debug_size=10
delete_weight=2
replace_weight=7
insert_weight=7
block_split_weight=6
select_method=bm25
model_type=codet5
pretrained_model=salesforce/codet5-large
load_locator_model_path=model/$lang/checkpoint-last/pytorch_model.bin

python run.py --lang $lang \
 --do_test --do_train --do_eval \
 --model_type $model_type --model_name_or_path $pretrained_model \
 --train_filename $train_file --dev_filename $dev_file --test_filename $test_file \
 --output_dir $output_dir \
 --max_source_length $source_length \
 --locator_batch_size $batch_size \
 --learning_rate $lr --num_train_epochs $epochs \
 --delete_weight $delete_weight --replace_weight $replace_weight \
 --insert_weight $insert_weight --block_split_weight $block_split_weight \
 --select_method $select_method \
 --label_num $label_num \
#  --debug_mode --debug_size $debug_size
