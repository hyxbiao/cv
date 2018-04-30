
input_file=${1}
attr_key=${2:-skirt_length_labels}
data_dir=${3:-~/data/vision/fashionAI/}
model_dir=${4:-models/$attr_key/baseline}

day=`date +%Y%m%d` 
now=`date +%Y%m%d_%H%M` 

output_dir=output/${attr_key}/$day
mkdir -p $output_dir

log_dir=log/$attr_key
mkdir -p $log_dir

echo "data_dir: $data_dir"
echo "attr_key: $attr_key"
echo "model_dir: $model_dir"
echo "output_dir: $output_dir"

python app.py \
	--data_dir $data_dir \
	--model_dir $model_dir \
	--attr_key $attr_key \
	--predict \
	--predict_input_file=$input_file
