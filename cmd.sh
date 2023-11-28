python -m scripts.generate_training_data --output_dir=data/pemsd7 --traffic_file=dataset/PeMSD7_V_228.csv
python dcrnn_train.py --config_filename=config/dcrnn_new.yaml
