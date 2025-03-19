system=CoEdPilot
locator_model_path=models/CoEdPilot/locator_model/pytorch_model.bin
invoker_model_path=None
generator_model_path=models/CoEdPilot/generator_model/pytorch_model.bin
dependency_model_path=models/CoEdPilot/dependency_model
estimator_model_path=models/CoEdPilot/estimator_model/pytorch_model.bin
locator_batch_size=20
output_dir=./output/$system

lang=$1
idx=$2

python run.py \
 --system $system \
 --locator_model_path $locator_model_path \
 --generator_model_path $generator_model_path \
 --dependency_model_path $dependency_model_path \
 --estimator_model_path $estimator_model_path \
 --locator_batch_size $locator_batch_size \
 --testset_path testset.json \
 --output_dir $output_dir \
 --random_order \
 --label_correction \
 --lang $lang \
 --idx $idx