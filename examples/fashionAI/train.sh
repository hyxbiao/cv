
data_dir=${1:-$HOME/data/vision/fashionAI/}
attr_key=${2:-skirt_length_labels}
pretrain_model_dir=${3:-models/pretrain_resnet50/}

base_model_dir=models/$attr_key
mkdir -p $base_model_dir

now=`date +%Y%m%d_%H%M` 
model_dir=$base_model_dir/$now

#pretrain_model_dir=models/skirt_length_labels/baseline/
pretrain_model_dir=models/pretrain_resnet50/
pretrain_warm_vars='^((?!dense).)*$'

#if [ "$attr_key" = "skirt_length_labels" ]; then
#	pretrain_warm_vars='.*'
#	pretrain_model_dir=model_skirt_pretrain/
#fi

mkdir log


echo "data_dir: $data_dir"
echo "attr_key: $attr_key"
echo "model_dir: $model_dir"
echo "pretrain_model_dir: $pretrain_model_dir"
echo "pretrain_warm_vars: $pretrain_warm_vars"

nohup python app.py \
	--data_dir $data_dir \
	--model_dir $model_dir \
	--attr_key $attr_key \
	--pretrain_model_dir $pretrain_model_dir \
	--pretrain_warm_vars $pretrain_warm_vars \
	>log/${attr_key}_${now}.out 2>&1 &
