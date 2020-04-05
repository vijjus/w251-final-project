from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/data/caption_data/dataset_coco.json',
                       image_folder='/data/caption_data/mscoco',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/data/caption_data/mscoco/caption_data/',
                       max_len=50)
