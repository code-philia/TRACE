lr=5e-5
batch_size=16
source_length=512
data_dir=dataset
output_dir=model
train_file=$data_dir/train.json
dev_file=$data_dir/dev.json
test_file=$data_dir/test.json
epochs=2
debug_size=10
model_type=roberta
pretrained_model=microsoft/codebert-base
load_model_path=model/checkpoint-last/pytorch_model.bin

python run.py \
 --do_train --do_eval \
 --model_type $model_type --model_name_or_path $pretrained_model \
 --train_filename $train_file --dev_filename $dev_file --test_filename $test_file \
 --output_dir $output_dir --dataset_dir $data_dir \
 --max_source_length $source_length \
 --batch_size $batch_size \
 --learning_rate $lr --num_train_epochs $epochs \
#  --debug_mode --debug_size $debug_size
