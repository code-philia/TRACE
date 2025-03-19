system=EnrichedSemantics
locator_model_path=models/Enriched/locator_model/pytorch_model.bin
invoker_model_path=None
generator_model_path=models/Enriched/generator_model/pytorch_model.bin
locator_batch_size=20
output_dir=./output/$system

lang=$1
idx=$2

python run.py \
 --system $system \
 --locator_model_path $locator_model_path \
 --invoker_model_path $invoker_model_path \
 --generator_model_path $generator_model_path \
 --locator_batch_size $locator_batch_size \
 --testset_path testset.json \
 --output_dir $output_dir \
 --random_order \
 --label_correction \
 --lang $lang \
 --idx $idx