# python code/score.py


result_path="results_backup/"
# result_path="results/"
labeled_data_path="data/"
saved_score_path="scores/"

mkdir -p ${saved_score_path}

python code/score.py --result_path ${result_path} --labeled_data_path ${labeled_data_path} --saved_score_path ${saved_score_path}

