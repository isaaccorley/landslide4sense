python predict.py --predict_on val \
    --output_directory outputs \
    --device cuda \
    --log_dir logs/version_0
zip submission.zip outputs/*.h5
